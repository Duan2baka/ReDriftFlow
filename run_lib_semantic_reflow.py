import torch
import numpy as np
import os
import json
import logging
from tqdm import tqdm
from typing import Optional, Tuple
import pickle
import sys
import gc
import wandb
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.utils as vutils

# Set memory management for PyTorch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Add ImageGeneration directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ImageGeneration'))

from semantic_boundary_trainer import SemanticBoundaryTrainer

# Import adaptive control strength functions
try:
    from uspace_train import get_adaptive_control_strength, determine_target_direction, print_k_statistics
except ImportError as e:
    logging.warning(f"Adaptive control functions not found: {e}")
    # Define fallback functions
    def get_adaptive_control_strength(probs, labels, base_strength=1.0):
        return base_strength
    def determine_target_direction(labels):
        return 1.0
    def print_k_statistics(k_values):
        pass

# Import ImageGeneration modules with try-except for development
try:
    from ImageGeneration import run_lib_reflow
    from ImageGeneration import sampling
    from ImageGeneration import sde_lib
    from ImageGeneration.models import utils as mutils
    from ImageGeneration.models.ema import ExponentialMovingAverage
    from ImageGeneration import datasets
    from ImageGeneration.utils import save_checkpoint
except ImportError as e:
    logging.warning(f"Some ImageGeneration modules not found: {e}")
    # For development/testing, define minimal stubs
    class MockModule:
        pass
    run_lib_reflow = MockModule()
    sampling = MockModule()
    sde_lib = MockModule()
    mutils = MockModule()
    ExponentialMovingAverage = MockModule()
    datasets = MockModule()
    save_checkpoint = MockModule()


def save_images_from_tensor(image_tensor, save_dir, prefix="image", start_idx=0):
    """Save images from tensor to individual files using PIL (same as generate_celeba.py)"""
    from PIL import Image
    import numpy as np
    
    os.makedirs(save_dir, exist_ok=True)
    
    if torch.is_tensor(image_tensor):
        images = image_tensor.cpu()
    else:
        images = torch.from_numpy(image_tensor)
    
    # Check the range of the input data
    min_val = images.min().item()
    max_val = images.max().item()
    
    # The images from sampling_fn should already be in [0,1] range after inverse_scaler
    # Only normalize if they are in [-1,1] range
    if min_val < -0.5:  # Likely in [-1,1] range
        images = (images + 1.0) / 2.0
    
    images = torch.clamp(images, 0, 1)
    
    # Convert to numpy and rescale to [0, 255] (same as generate_celeba.py)
    images_np = images.permute(0, 2, 3, 1).numpy()  # NCHW → NHWC
    images_np = np.clip(images_np * 255.0, 0, 255).astype(np.uint8)
    
    saved_paths = []
    for i, img in enumerate(images_np):
        img_path = os.path.join(save_dir, f"{prefix}_{start_idx + i:06d}.png")
        img_pil = Image.fromarray(img)  # Same as generate_celeba.py
        img_pil.save(img_path)
        saved_paths.append(img_path)
    
    return saved_paths

def create_image_grid(images, nrow=8, save_path=None):
    """Create and save a grid of images using matplotlib (same style as generate_celeba.py)"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    if torch.is_tensor(images):
        img_tensor = images.cpu()
    else:
        img_tensor = torch.from_numpy(images)
    
    min_val = img_tensor.min().item()
    max_val = img_tensor.max().item()
    
    if min_val < -0.5:  # Likely in [-1,1] range
        img_tensor = (img_tensor + 1.0) / 2.0
    
    img_tensor = torch.clamp(img_tensor, 0, 1)
    
    # Convert to numpy format for matplotlib (NHWC)
    images_np = img_tensor.permute(0, 2, 3, 1).numpy()
    images_np = np.clip(images_np * 255.0, 0, 255).astype(np.uint8)
    
    # Create grid layout
    num_images = len(images_np)
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))
    
    if grid_size == 1:
        axes = [axes]
    elif grid_size > 1:
        axes = axes.flatten()
    
    for i in range(grid_size * grid_size):
        if i < num_images:
            axes[i].imshow(images_np[i])
            axes[i].axis('off')
        else:
            axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
        return fig


def setup_wandb(config, workdir, project_name="semantic-reflow"):
    """Initialize WandB logging"""
    wandb.init(
        project=project_name,
        config={
            "sample_steps": config.sampling.sample_N,
            "batch_size": config.semantic.semantic_batch_size,
            "num_samples": getattr(config.semantic, 'num_semantic_samples', 50000),
            "image_size": config.data.image_size,
            "ode_tol": config.sampling.ode_tol,
            "workdir": workdir,
        },
        dir=workdir
    )
    return wandb.run

def log_data_pairs_examples(z0_data, z1_data, num_examples=8):
    """Log examples of generated data pairs to WandB"""
    if len(z0_data) == 0 or len(z1_data) == 0:
        return
    
    num_examples = min(num_examples, len(z0_data))
    
    # Convert images from [-1,1] to [0,1] for visualization
    images = []
    for i in range(num_examples):
        img = z1_data[i]
        if torch.is_tensor(img):
            img = img.cpu().numpy()
        
        # Normalize to [0,1]
        if img.min() < 0:
            img = (img + 1.0) / 2.0
        img = np.clip(img, 0, 1)
        
        # Convert from CHW to HWC if needed
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        
        images.append(wandb.Image(img, caption=f"Generated Image {i+1}"))
    
    wandb.log({"generated_data_examples": images})

def log_classification_results(labels, boundary_accuracy=None):
    """Log semantic classification statistics"""
    labels = np.array(labels)
    num_positive = np.sum(labels == 1)
    num_negative = np.sum(labels == 0)
    total = len(labels)
    
    classification_stats = {
        "classification/num_positive": num_positive,
        "classification/num_negative": num_negative,
        "classification/total_samples": total,
        "classification/positive_ratio": num_positive / total if total > 0 else 0,
    }
    
    if boundary_accuracy is not None:
        classification_stats["classification/svm_accuracy"] = boundary_accuracy
    
    wandb.log(classification_stats)
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=(8, 6))
    labels_pie = ['Positive (Smile)', 'Negative (No Smile)']
    sizes = [num_positive, num_negative]
    colors = ['#ff9999', '#66b3ff']
    
    ax.pie(sizes, labels=labels_pie, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Classification Distribution')
    
    wandb.log({"classification/distribution_chart": wandb.Image(fig)})
    plt.close(fig)

def log_boundary_training_results(boundary_details):
    """Log semantic boundary training results"""
    if boundary_details is None:
        return
    
    boundary_stats = {
        "boundary/training_samples": boundary_details.get('training_samples', 0),
        "boundary/positive_samples": boundary_details.get('positive_samples', 0),
        "boundary/negative_samples": boundary_details.get('negative_samples', 0),
        "boundary/vector_norm": boundary_details.get('boundary_norm', 0),
    }
    
    wandb.log(boundary_stats)


def load_model_and_config(config, checkpoint_path):
    """Load the trained model and configuration following generate_celeba.py pattern"""
    
    device = config.device
    
    # Create data scaler
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    
    # Initialize model
    score_model = mutils.create_model(config)
    optimizer = torch.optim.Adam(score_model.parameters())  # Dummy optimizer for loading
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    
    # Create state dict
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    
    # Load checkpoint
    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    state['step'] = checkpoint['step']
    state['optimizer'].load_state_dict(checkpoint['optimizer'])
    state['model'].load_state_dict(checkpoint['model'], strict=False)
    state['ema'].load_state_dict(checkpoint['ema'])
    
    # Copy EMA parameters to model
    ema.copy_to(score_model.parameters())
    score_model.to(device)  # Move model to GPU
    score_model.eval()
    
    logging.info(f"Model loaded successfully! Training step: {state['step']}")
    logging.info(f"Model moved to device: {device}")
    
    return score_model, inverse_scaler

def generate_data_from_z0(config, workdir, num_samples):
    """Generate data from random noise using pre-trained RectifiedFlow model."""
    logging.info(f"Generating {num_samples} data pairs...")
    
    # Clear GPU cache before starting
    torch.cuda.empty_cache()
    
    # Create output directories
    generate_dir = os.path.join(workdir, 'generated_data')
    intermediate_dir = os.path.join(workdir, 'intermediate_results')
    images_dir = os.path.join(intermediate_dir, 'images')
    grids_dir = os.path.join(intermediate_dir, 'grids')
    pt_files_dir = os.path.join(intermediate_dir, 'pt_files')
    
    os.makedirs(generate_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(grids_dir, exist_ok=True)
    os.makedirs(pt_files_dir, exist_ok=True)
    
    # Initialize SDE using correct parameter names from config
    sde = sde_lib.RectifiedFlow(
        init_type=config.sampling.init_type,
        noise_scale=config.sampling.init_noise_scale,
        use_ode_sampler=config.sampling.use_ode_sampler,
        sigma_var=getattr(config.sampling, 'sigma_variance', 0.0),
        ode_tol=getattr(config.sampling, 'ode_tol', 1e-5),
        sample_N=getattr(config.sampling, 'sample_N', 50)
    )
    
    checkpoint_path = config.reflow.last_flow_ckpt
    score_model, inverse_scaler = load_model_and_config(config, checkpoint_path)
    
    max_batch_size = min(config.semantic.semantic_batch_size, 4)  # Max 8 samples at a time
    generation_batch_size = max_batch_size
    
    sampling_shape = (generation_batch_size, config.data.num_channels, 
                     config.data.image_size, config.data.image_size)
    sampling_eps = 1e-3
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
    
    num_batches = (num_samples + generation_batch_size - 1) // generation_batch_size
    
    # Store batch file paths instead of keeping data in memory
    batch_files = []
    
    for batch_idx in tqdm(range(num_batches), desc="Generating data pairs"):
        # Clear cache at start of each batch
        torch.cuda.empty_cache()
        
        # Calculate actual batch size for this iteration
        remaining_samples = num_samples - batch_idx * generation_batch_size
        current_batch_size = min(generation_batch_size, remaining_samples)
        
        if current_batch_size != generation_batch_size:
            # Update sampling shape for the last batch
            sampling_shape = (current_batch_size, config.data.num_channels, 
                             config.data.image_size, config.data.image_size)
            sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
        
        # Sample initial noise
        z0 = sde.get_z0(torch.zeros(sampling_shape, device=config.device), 
                       train=False).to(config.device)
        
        # Generate samples
        with torch.no_grad():
            z1, _ = sampling_fn(score_model)
        
        # Save each batch immediately to avoid memory accumulation
        batch_file_path = os.path.join(pt_files_dir, f'batch_{batch_idx+1}_pairs.pt')
        
        # Move to CPU and ensure contiguous memory layout
        z0_cpu = z0.cpu().contiguous()
        z1_cpu = z1.cpu().contiguous()
        
        torch.save({
            'z0_batch': z0_cpu,
            'z1_batch': z1_cpu,
            'batch_idx': batch_idx,
            'total_batches': num_batches,
            'batch_size': z0.shape[0]
        }, batch_file_path, pickle_protocol=4, _use_new_zipfile_serialization=False)

        # Clean up CPU tensors
        del z0_cpu, z1_cpu
        
        # Save images as individual files
        batch_start_idx = batch_idx * generation_batch_size
        saved_image_paths = save_images_from_tensor(
            z1.cpu(), 
            images_dir, 
            prefix=f"generated", 
            start_idx=batch_start_idx
        )
        
        # Create and save batch grid every 20 batches or at the end
        if (batch_idx + 1) % 20 == 0 or batch_idx == num_batches - 1:
            grid_path = os.path.join(grids_dir, f'batch_{batch_idx+1}_grid.png')
            create_image_grid(z1.cpu(), nrow=min(4, z1.shape[0]), save_path=grid_path)
        
        batch_files.append(batch_file_path)
        
        # Log progress every 100 batches or at specific milestones to keep tqdm clean
        if (batch_idx + 1) % 1000 == 0 or batch_idx == num_batches - 1 or (batch_idx + 1) in [1, 10, 50]:
            current_samples = min((batch_idx + 1) * generation_batch_size, num_samples)
            logging.info(f"Progress: {batch_idx+1}/{num_batches} batches completed ({current_samples}/{num_samples} samples)")
        
        # Clear GPU memory immediately and force garbage collection
        del z0, z1
        torch.cuda.empty_cache()
        gc.collect()
    
    # Load and concatenate results from saved files
    logging.info("Loading and concatenating batch results...")
    all_z0 = []
    all_z1 = []
    
    # Process files in small chunks to avoid memory issues
    chunk_size = 10  # Load 10 batch files at a time
    for chunk_start in tqdm(range(0, len(batch_files), chunk_size), desc="Loading batch files"):
        chunk_end = min(chunk_start + chunk_size, len(batch_files))
        chunk_z0 = []
        chunk_z1 = []
        
        for i in range(chunk_start, chunk_end):
            # Handle both .pt and .npz files
            if batch_files[i].endswith('.npz'):
                import numpy as np
                batch_data = np.load(batch_files[i])
                chunk_z0.append(torch.from_numpy(batch_data['z0_batch']))
                chunk_z1.append(torch.from_numpy(batch_data['z1_batch']))
            else:
                batch_data = torch.load(batch_files[i], map_location='cpu')
                chunk_z0.append(batch_data['z0_batch'])
                chunk_z1.append(batch_data['z1_batch'])
        
        all_z0.append(torch.cat(chunk_z0, dim=0))
        all_z1.append(torch.cat(chunk_z1, dim=0))
        
        # Clear chunk data
        del chunk_z0, chunk_z1
        gc.collect()
    
    # Concatenate and trim to exact number requested
    z0_data = torch.cat(all_z0, dim=0)[:num_samples]
    z1_data = torch.cat(all_z1, dim=0)[:num_samples]
    
    # Clear intermediate lists to free memory
    del all_z0, all_z1
    gc.collect()
    
    # Save final consolidated data with robust error handling
    data_path = os.path.join(generate_dir, 'generated_pairs.pt')
    z0_save = z0_data.cpu().contiguous()
    z1_save = z1_data.cpu().contiguous()
    
    torch.save({'z0': z0_save, 'z1': z1_save}, data_path, pickle_protocol=4, _use_new_zipfile_serialization=False)
    logging.info(f"Saved final data to {data_path}")
        
    
    # Clean up data copies
    if 'z0_save' in locals():
        del z0_save
    if 'z1_save' in locals():
        del z1_save
    
    # Save final image grid
    final_grid_path = os.path.join(generate_dir, 'all_generated_grid.png')
    sample_images = z1_data[:min(64, len(z1_data))]  # Sample up to 64 images for grid
    create_image_grid(sample_images, nrow=8, save_path=final_grid_path)
    logging.info(f"Saved final image grid to {final_grid_path}")
    
    # Log examples to WandB
    log_data_pairs_examples(z0_data, z1_data, num_examples=8)
    
    # Save batch file index for potential recovery
    batch_index_path = os.path.join(pt_files_dir, 'batch_file_index.pkl')
    batch_index = {
        'batch_files': batch_files,
        'total_samples': num_samples,
        'num_batches': num_batches,
        'generation_batch_size': generation_batch_size
    }
    
    with open(batch_index_path, 'wb') as f:
        pickle.dump(batch_index, f)
    
    logging.info(f"Saved batch file index to {batch_index_path}")
    logging.info(f"Generated {len(z0_data)} data pairs, saved to: {data_path}")
    print(f"Generated {len(z0_data)} data pairs, saved to: {data_path}")
    
    # Clean up memory before returning
    del z0_data, z1_data
    torch.cuda.empty_cache()
    gc.collect()
    
    return data_path

def generate_semantic_data_pairs(config, 
                                workdir: str,
                                interfacegan_model_path: str,
                                num_samples: int = 10000) -> str:
    """
    Generate semantic classified (noise, data) pairs
    Returns path to generated data files
    """
    logging.info(f"Generating {num_samples} semantic data pairs...")
    
    # Create semantic analysis directories
    semantic_dir = os.path.join(workdir, 'semantic_analysis')
    classification_dir = os.path.join(semantic_dir, 'classifications')
    pt_semantic_dir = os.path.join(semantic_dir, 'pt_files')
    classified_images_dir = os.path.join(semantic_dir, 'classified_images')
    
    for dir_path in [semantic_dir, classification_dir, pt_semantic_dir, classified_images_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Initialize semantic boundary trainer
    semantic_trainer = SemanticBoundaryTrainer(
        interfacegan_model_path=interfacegan_model_path,
        device=config.device
    )
    
    # Generate data pairs
    _ = generate_data_from_z0(config, workdir, num_samples)
    
    # Load batch file index
    pt_files_dir = os.path.join(workdir, 'intermediate_results', 'pt_files')
    batch_index_path = os.path.join(pt_files_dir, 'batch_file_index.pkl')
    
    with open(batch_index_path, 'rb') as f:
        batch_index = pickle.load(f)
    
    batch_files = batch_index['batch_files']
    
    # Process semantic classification
    all_noise_files = []
    all_images_files = []
    all_labels = []
    positive_count = 0
    negative_count = 0
    
    for batch_idx, batch_file_path in enumerate(tqdm(batch_files, desc="Classifying images")):
        # Load single batch
        batch_data = torch.load(batch_file_path, map_location='cpu')
        z0_batch = batch_data['z0_batch']
        z1_batch = batch_data['z1_batch']
        
        z1_tensor = z1_batch.to(config.device)
        
        if z1_tensor.shape[-1] == 3:
            z1_tensor = z1_tensor.permute(0, 3, 1, 2)
        
        if z1_tensor.min() < -0.5:
            z1_tensor = (z1_tensor + 1.0) / 2.0
        
        z1_tensor = torch.clamp(z1_tensor, 0, 1)
        
        labels, probabilities = semantic_trainer.classify_images(
            z1_tensor, 
            debug=(batch_idx == 0), 
            return_probabilities=True
        )
        
        batch_num = batch_idx + 1
        batch_noise_path = os.path.join(pt_semantic_dir, f'noise_batch_{batch_num}.pt')
        batch_images_path = os.path.join(pt_semantic_dir, f'images_batch_{batch_num}.pt')
        batch_probs_path = os.path.join(pt_semantic_dir, f'probabilities_batch_{batch_num}.pt')
        
        torch.save(z0_batch.contiguous(), batch_noise_path, pickle_protocol=4, _use_new_zipfile_serialization=False)
        torch.save(z1_batch.contiguous(), batch_images_path, pickle_protocol=4, _use_new_zipfile_serialization=False)
        torch.save(torch.tensor(probabilities), batch_probs_path, pickle_protocol=4, _use_new_zipfile_serialization=False)
        #print(f"Batch {batch_num} - Noise shape: {z0_batch.shape}, Image shape: {z1_batch.shape}")
        #print(f"Loaded noise dtype: {z0_batch.dtype}")
        #print(f"Expected size: {z0_batch.numel() * z0_batch.element_size() / 1024 / 1024:.1f}MB")
        # Count classification results
        for label in labels:
            if label == 1:
                positive_count += 1
            else:
                negative_count += 1
        
        # Save debug samples (first batch only)
        if batch_idx == 0:
            sample_dir = os.path.join(classified_images_dir, 'debug_samples')
            os.makedirs(sample_dir, exist_ok=True)
            
            for j in range(min(4, len(z1_tensor))):
                img = z1_tensor[j].cpu()
                label = labels[j]
                
                img_normalized = torch.clamp(img, 0, 1)
                img_np = img_normalized.permute(1, 2, 0).numpy()
                img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                
                label_name = "smile" if label == 1 else "no_smile"
                img_path = os.path.join(sample_dir, f'debug_{j}_{label_name}.png')
                img_pil.save(img_path)
        
        all_noise_files.append(batch_noise_path)
        all_images_files.append(batch_images_path)
        all_labels.extend(labels)
        
        # Create classification grids every 50 batches
        if batch_num % 50 == 0:
            positive_indices = [i for i, label in enumerate(labels) if label == 1]
            negative_indices = [i for i, label in enumerate(labels) if label == 0]
            
            if positive_indices:
                pos_samples = z1_tensor[positive_indices[:16]]
                pos_grid_path = os.path.join(classified_images_dir, f'positive_grid_batch_{batch_num}.png')
                create_image_grid(pos_samples.cpu(), nrow=4, save_path=pos_grid_path)
            
            if negative_indices:
                neg_samples = z1_tensor[negative_indices[:16]]
                neg_grid_path = os.path.join(classified_images_dir, f'negative_grid_batch_{batch_num}.png')
                create_image_grid(neg_samples.cpu(), nrow=4, save_path=neg_grid_path)
        
        # Clean up memory
        del z0_batch, z1_batch, z1_tensor
        torch.cuda.empty_cache()
        
        if batch_num % 10 == 0:
            gc.collect()
    
    # Save file index
    file_index_path = os.path.join(pt_semantic_dir, 'batch_files_index.pkl')
    file_index = {
        'noise_files': all_noise_files,
        'images_files': all_images_files,
        'labels': all_labels,
        'total_batches': len(all_noise_files)
    }
    
    with open(file_index_path, 'wb') as f:
        pickle.dump(file_index, f)
    
    # Save classification summary
    classification_summary = {
        'num_positive': positive_count,
        'num_negative': negative_count,
        'total_samples': positive_count + negative_count,
        'file_index_path': file_index_path
    }
    
    summary_path = os.path.join(classification_dir, 'classification_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(classification_summary, f, indent=2)
    
    logging.info(f"Classification completed: {positive_count} positive, {negative_count} negative")
    logging.info(f"Saved classification summary to {summary_path}")
    
    return classification_dir


def train_reflow_with_uspace_drift(config, workdir, z0_data, z1_data, labels, boundary_vector, probabilities=None):
    """
    Train reflow with u-space drift at t=0.25
    在训练时对 zt 添加语义控制偏移: zt' = zt + k*v
    损失函数还是 L(z1, z_hat)，但UNet的输入被修改了
    
    Args:
        probabilities: Classifier confidence probabilities for adaptive control strength
    """
    logging.info("Training reflow with u-space drift modification...")
    
    # Calculate adaptive control strengths if probabilities are available
    k_values = None
    if probabilities is not None:
        logging.info("Computing adaptive control strengths based on classifier confidence...")
        
        # Get base strength from config or use default
        base_strength = config.get('semantic_control', {}).get('base_strength', 1.0)
        
        # Calculate adaptive strengths for each sample
        k_values = []
        target_directions = []
        
        # For semantic control, typically we want to enhance the attribute (move towards positive)
        desired_label = 1  # Enhance attribute by default, could be configurable
        
        for i, (prob, label) in enumerate(zip(probabilities, labels)):
            # Get target direction based on current and desired labels
            direction = determine_target_direction(label, desired_label)
            
            # Get adaptive strength based on confidence and direction
            k = get_adaptive_control_strength(prob, direction, base_strength=base_strength)
            
            k_values.append(k)
            target_directions.append(direction)
        
        k_values = np.array(k_values)
        target_directions = np.array(target_directions)
        
        # Log adaptive strength statistics
        print_k_statistics(k_values, probabilities, epoch=0)
        
        # Log to WandB
        wandb.log({
            "adaptive_control/mean_strength": np.mean(k_values),
            "adaptive_control/std_strength": np.std(k_values),
            "adaptive_control/min_strength": np.min(k_values),
            "adaptive_control/max_strength": np.max(k_values),
            "adaptive_control/num_positive_direction": np.sum(target_directions > 0),
            "adaptive_control/num_negative_direction": np.sum(target_directions < 0)
        })
        
        logging.info(f"Adaptive strengths: mean={np.mean(k_values):.3f}, std={np.std(k_values):.3f}")
    else:
        logging.info("No probabilities provided, using fixed control strength")
        base_strength = config.get('semantic_control', {}).get('base_strength', 1.0)
        k_values = np.full(len(labels), base_strength)
        desired_label = 1  # Default to enhance attribute
        target_directions = [determine_target_direction(label, desired_label) for label in labels]
    
    # Create checkpoint directory for training
    training_checkpoint_dir = os.path.join(workdir, 'semantic_training', 'checkpoints')
    os.makedirs(training_checkpoint_dir, exist_ok=True)
    
    # Update workdir to point to semantic_training so checkpoints are saved there
    semantic_training_workdir = os.path.join(workdir, 'semantic_training')
    os.makedirs(semantic_training_workdir, exist_ok=True)
    
    # Prepare boundary vector
    if isinstance(boundary_vector, np.ndarray):
        boundary_tensor = torch.from_numpy(boundary_vector)
    else:
        boundary_tensor = boundary_vector
    
    # Reshape boundary to image dimensions if it's flattened
    if len(boundary_tensor.shape) == 1:
        boundary_tensor = boundary_tensor.reshape(config.data.num_channels, 
                                                config.data.image_size, 
                                                config.data.image_size)
    
    # Save training data with semantic control info
    training_data = {
        'noise': z0_data,
        'data': z1_data,  # Original z1, no modification needed
        'labels': labels,
        'probabilities': probabilities,
        'boundary_vector': boundary_vector,
        'adaptive_strengths': k_values,
        'target_directions': target_directions,
        'control_time': 0.25,  # t=0.25时刻应用控制
        'metadata': {
            'total_samples': len(z0_data),
            'boundary_norm': np.linalg.norm(boundary_vector) if boundary_vector is not None else None,
            'training_type': 'uspace_drift_adaptive' if probabilities is not None else 'uspace_drift_fixed',
            'adaptive_control': probabilities is not None,
            'mean_strength': np.mean(k_values) if k_values is not None else None,
            'std_strength': np.std(k_values) if k_values is not None else None
        }
    }
    
    # Save training data
    training_data_path = os.path.join(semantic_training_workdir, 'uspace_drift_training_data.pkl')
    with open(training_data_path, 'wb') as f:
        pickle.dump(training_data, f)
    
    logging.info(f"Saved u-space drift training data to {training_data_path}")
    logging.info(f"Training data: {len(z0_data)} samples with u-space drift at t=0.25")
    
    # Update config to use the new training data
    config.data.dataset = 'CUSTOM'
    config.data.data_path = training_data_path
    
    try:
        # 使用自定义训练函数，在训练时对zt应用u-space drift
        custom_reflow_training_with_uspace_drift(config, semantic_training_workdir, training_data_path)
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise
    
    logging.info("U-space drift reflow training completed!")
    logging.info(f"Model checkpoints saved in: {training_checkpoint_dir}")
    
    return training_checkpoint_dir

def train_reflow(config, workdir):
    """
    Train reflow model using the standard reflow training pipeline with checkpoint saving
    """
    logging.info("Starting reflow training with checkpoint management...")
    
    # Create checkpoint directory for training
    training_checkpoint_dir = os.path.join(workdir, 'semantic_training', 'checkpoints')
    os.makedirs(training_checkpoint_dir, exist_ok=True)
    
    # Update workdir to point to semantic_training so checkpoints are saved there
    semantic_training_workdir = os.path.join(workdir, 'semantic_training')
    os.makedirs(semantic_training_workdir, exist_ok=True)
    
    try:
        # Use the existing finetune_reflow function with modified workdir
        run_lib_reflow.finetune_reflow(config, semantic_training_workdir)
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise
    
    logging.info("Reflow training completed!")
    logging.info(f"Model checkpoints saved in: {training_checkpoint_dir}")
    
    # Return the checkpoint directory for the caller
    return training_checkpoint_dir

def train_semantic_reflow(config, workdir: str, semantic_data_path: str):
    """
    Train reflow model using semantically-aware data pairs
    """
    logging.info("Training semantic-aware reflow model...")
    
    # Create training analysis directories
    training_dir = os.path.join(workdir, 'semantic_training')
    checkpoint_dir = os.path.join(training_dir, 'checkpoints')
    progress_dir = os.path.join(training_dir, 'progress')
    
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(progress_dir, exist_ok=True)
    
    # Load semantic data (only metadata first to check size)
    logging.info("Loading semantic data metadata...")
    with open(semantic_data_path, 'rb') as f:
        semantic_data = pickle.load(f)
    
    # Check if we need to load from intermediate files or use existing data
    if 'metadata' in semantic_data and 'classification_dir' in semantic_data['metadata']:
        # Load data from intermediate files to save memory
        logging.info("Loading data from intermediate files to manage memory...")
        
        file_index_path = os.path.join(workdir, 'semantic_analysis', 'pt_files', 'batch_files_index.pkl')
        if os.path.exists(file_index_path):
            with open(file_index_path, 'rb') as f:
                file_index = pickle.load(f)
            
            # Load data in chunks
            z0_data_chunks = []
            z1_data_chunks = []
            labels = file_index['labels']
            probabilities = file_index.get('probabilities', None)
            
            chunk_size = 3  # Load 3 batches at a time
            for i in range(0, len(file_index['noise_files']), chunk_size):
                chunk_end = min(i + chunk_size, len(file_index['noise_files']))
                
                chunk_z0 = []
                chunk_z1 = []
                
                for j in range(i, chunk_end):
                    z0_batch = torch.load(file_index['noise_files'][j], map_location='cpu')
                    z1_batch = torch.load(file_index['images_files'][j], map_location='cpu')
                    chunk_z0.append(z0_batch)
                    chunk_z1.append(z1_batch)
                
                z0_data_chunks.append(torch.cat(chunk_z0, dim=0))
                z1_data_chunks.append(torch.cat(chunk_z1, dim=0))
                
                del chunk_z0, chunk_z1
            
            z0_data = torch.cat(z0_data_chunks, dim=0)
            z1_data = torch.cat(z1_data_chunks, dim=0)
            labels = np.array(labels)
            if probabilities is not None:
                probabilities = np.array(probabilities)
            
            del z0_data_chunks, z1_data_chunks
        else:
            # Fallback to loading from semantic_data
            z0_data = semantic_data['z0']
            z1_data = semantic_data['z1']
            labels = semantic_data['labels']
            probabilities = semantic_data.get('probabilities', None)
    else:
        # Extract data from semantic data
        z0_data = semantic_data['z0']
        z1_data = semantic_data['z1']
        labels = semantic_data['labels']
        probabilities = semantic_data.get('probabilities', None)
    
    # Save original data statistics before processing
    original_stats_path = os.path.join(progress_dir, 'original_data_stats.pkl')
    original_stats = {
        'z0_shape': z0_data.shape if hasattr(z0_data, 'shape') else None,
        'z1_shape': z1_data.shape if hasattr(z1_data, 'shape') else None,
        'labels_shape': labels.shape if hasattr(labels, 'shape') else None,
        'positive_count': np.sum(labels == 1) if hasattr(labels, '__len__') else 0,
        'negative_count': np.sum(labels == 0) if hasattr(labels, '__len__') else 0,
        'total_samples': len(labels) if hasattr(labels, '__len__') else 0
    }
    
    with open(original_stats_path, 'wb') as f:
        pickle.dump(original_stats, f)
    
    logging.info(f"Saved original data statistics to {original_stats_path}")
    
    # Convert numpy arrays to tensors if needed
    if isinstance(z0_data, np.ndarray):
        z0_data = torch.from_numpy(z0_data)
    if isinstance(z1_data, np.ndarray):
        z1_data = torch.from_numpy(z1_data)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    
    # Optional: Balance the dataset
    if config.get('balance_classes', False):
        pos_indices = torch.where(labels == 1)[0]
        neg_indices = torch.where(labels == 0)[0]
        
        min_class_size = min(len(pos_indices), len(neg_indices))
        
        balanced_indices = torch.cat([
            pos_indices[torch.randperm(len(pos_indices))[:min_class_size]],
            neg_indices[torch.randperm(len(neg_indices))[:min_class_size]]
        ])
        
        z0_data = z0_data[balanced_indices]
        z1_data = z1_data[balanced_indices]
        labels = labels[balanced_indices]
        if probabilities is not None:
            if isinstance(probabilities, np.ndarray):
                probabilities = torch.from_numpy(probabilities)
            probabilities = probabilities[balanced_indices]
        
        logging.info(f"Balanced dataset: {len(z0_data)} samples ({torch.sum(labels == 1)} positive, {torch.sum(labels == 0)} negative)")
        
        # Save balanced data statistics
        balanced_stats_path = os.path.join(progress_dir, 'balanced_data_stats.pkl')
        balanced_stats = {
            'balanced_indices': balanced_indices.numpy(),
            'min_class_size': min_class_size,
            'final_positive_count': torch.sum(labels == 1).item(),
            'final_negative_count': torch.sum(labels == 0).item(),
            'final_total_samples': len(z0_data)
        }
        
        with open(balanced_stats_path, 'wb') as f:
            pickle.dump(balanced_stats, f)
        
        logging.info(f"Saved balanced data statistics to {balanced_stats_path}")
    
    # Save balanced data for training in a format compatible with reflow
    balanced_data_path = os.path.join(training_dir, 'balanced_semantic_pairs.pkl')
    
    # Ensure data is in correct format and range
    # z0 should be noise, z1 should be images in [-1, 1] range
    if z1_data.max() > 1.0:
        # If images are in [0, 255] range, normalize to [-1, 1]
        z1_normalized = (z1_data / 255.0 - 0.5) / 0.5
    else:
        # If images are in [0, 1] range, normalize to [-1, 1]
        z1_normalized = (z1_data - 0.5) / 0.5
    
    # For noise, ensure it's properly normalized
    if hasattr(config.sampling, 'init_type') and config.sampling.init_type == 'gaussian':
        z0_normalized = z0_data  # Keep Gaussian noise as is
    else:
        z0_normalized = (z0_data - 0.5) / 0.5  # Normalize to [-1, 1] if needed
    
    balanced_data = {
        'data': z1_normalized.numpy() if torch.is_tensor(z1_normalized) else z1_normalized,
        'noise': z0_normalized.numpy() if torch.is_tensor(z0_normalized) else z0_normalized,
        'labels': labels.numpy() if torch.is_tensor(labels) else labels,
        'probabilities': probabilities.numpy() if torch.is_tensor(probabilities) and probabilities is not None else probabilities
    }
    
    with open(balanced_data_path, 'wb') as f:
        pickle.dump(balanced_data, f)
    
    # Clear large data from memory after saving
    del z0_data, z1_data, z0_normalized, z1_normalized
    torch.cuda.empty_cache()
    
    # Save normalization details
    normalization_details_path = os.path.join(progress_dir, 'normalization_details.pkl')
    normalization_details = {
        'balanced_data_path': balanced_data_path,
        'normalization_method': 'to_[-1,1]' if hasattr(z1_data, 'max') and z1_data.max() <= 1.0 else 'from_[0,255]_to_[-1,1]',
        'final_sample_count': len(labels)
    }
    
    with open(normalization_details_path, 'wb') as f:
        pickle.dump(normalization_details, f)
    
    logging.info(f"Balanced semantic data saved to {balanced_data_path}")
    logging.info(f"Normalization details saved to {normalization_details_path}")
    
    # Update config to use custom dataset
    config.data.dataset = 'CUSTOM'
    config.data.data_path = balanced_data_path
    
    # Save training configuration
    training_config_path = os.path.join(training_dir, 'training_config.pkl')
    training_config = {
        'original_config': config,
        'semantic_data_path': semantic_data_path,
        'balanced_data_path': balanced_data_path,
        'checkpoint_dir': checkpoint_dir,
        'progress_dir': progress_dir
    }
    
    with open(training_config_path, 'wb') as f:
        pickle.dump(training_config, f)
    
    logging.info(f"Training configuration saved to {training_config_path}")
    
    # Clear remaining data from memory before training
    del labels
    if 'semantic_data' in locals():
        del semantic_data
    torch.cuda.empty_cache()
    
    # Train the reflow model and get checkpoint directory
    checkpoint_dir = train_reflow(config, workdir)
    
    # Return the checkpoint directory so caller can find the latest checkpoint
    return checkpoint_dir
