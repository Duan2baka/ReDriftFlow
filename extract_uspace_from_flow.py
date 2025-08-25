#!/usr/bin/env python3

import os
import sys
import logging
import numpy as np
import torch
import pickle
import gc
from tqdm import tqdm
from typing import Tuple, Dict, List

# Set memory management for PyTorch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Add ImageGeneration directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ImageGeneration'))

from semantic_boundary_trainer import SemanticBoundaryTrainer

# Import ImageGeneration modules
try:
    from ImageGeneration import sampling
    from ImageGeneration import sde_lib
    from ImageGeneration import datasets
    from ImageGeneration.models import utils as mutils
    from ImageGeneration.models.ema import ExponentialMovingAverage
    # Import all model classes to register them
    from ImageGeneration.models import ncsnpp
    from ImageGeneration.models import ncsnv2
    logging.info("Successfully imported ImageGeneration modules")
except ImportError as e:
    logging.warning(f"Some ImageGeneration modules not found: {e}")
    # Add fallback path and try again
    sys.path.insert(0, '/home/felix/RectifiedFlow/ImageGeneration')
    try:
        import sampling
        import sde_lib
        import datasets
        from models import utils as mutils
        from models.ema import ExponentialMovingAverage
        from models import ncsnpp
        from models import ncsnv2
        logging.info("Successfully imported ImageGeneration modules via fallback")
    except ImportError as e2:
        logging.error(f"Failed to import ImageGeneration modules: {e2}")
        raise


class USpaceExtractor:
    """
    Extract U-Space representations from UNet bottleneck features during flow matching
    
    Given (z0, z1) pairs, extract UNet bottleneck features at interpolated points:
    - Interpolation: xt = (1-t) * z0 + t * z1
    - Extract bottleneck features from UNet as U-Space
    """
    
    def __init__(self, config, control_time: float = 0.25):
        self.config = config
        self.control_time = control_time
        self.device = config.device
        
        # Ensure model classes are imported and registered
        self._ensure_models_imported()
        
        # Initialize SDE for flow matching
        self.sde = sde_lib.RectifiedFlow(
            init_type=config.sampling.init_type,
            noise_scale=config.sampling.init_noise_scale,
            use_ode_sampler=config.sampling.use_ode_sampler,
            sigma_var=getattr(config.sampling, 'sigma_variance', 0.0),
            ode_tol=getattr(config.sampling, 'ode_tol', 1e-5),
            sample_N=getattr(config.sampling, 'sample_N', 50)
        )
        
        self.score_model = None
        self.inverse_scaler = None
        
    def _ensure_models_imported(self):
        """Ensure all model classes are imported and registered"""
        try:
            from ImageGeneration.models import ncsnpp
            from ImageGeneration.models import ncsnv2
            logging.info("Model classes imported successfully")
        except ImportError:
            try:
                import sys
                sys.path.insert(0, '/home/felix/RectifiedFlow/ImageGeneration')
                from models import ncsnpp
                from models import ncsnv2
                logging.info("Model classes imported via fallback")
            except ImportError as e:
                logging.error(f"Failed to import model classes: {e}")
                raise
        
    def load_model(self, checkpoint_path: str):
        """Load pre-trained RectifiedFlow model"""
        logging.info(f"Loading model from {checkpoint_path}")
        
        # Create data scaler
        scaler = datasets.get_data_scaler(self.config)
        self.inverse_scaler = datasets.get_data_inverse_scaler(self.config)
        
        # Initialize model
        logging.info(f"Creating model with name: {self.config.model.name}")
        logging.info(f"Available models: {list(mutils._MODELS.keys())}")
        self.score_model = mutils.create_model(self.config)
        optimizer = torch.optim.Adam(self.score_model.parameters())
        ema = ExponentialMovingAverage(self.score_model.parameters(), decay=self.config.model.ema_rate)
        
        # Create state dict
        state = dict(optimizer=optimizer, model=self.score_model, ema=ema, step=0)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state['step'] = checkpoint['step']
        state['optimizer'].load_state_dict(checkpoint['optimizer'])
        state['model'].load_state_dict(checkpoint['model'], strict=False)
        state['ema'].load_state_dict(checkpoint['ema'])
        
        # Copy EMA parameters to model
        ema.copy_to(self.score_model.parameters())
        self.score_model.to(self.device)  # Move model to GPU
        self.score_model.eval()
        
        logging.info(f"Model loaded successfully! Training step: {state['step']}")
        logging.info(f"Model moved to device: {self.device}")
        
    def extract_uspace_from_unet(self, z0_data: np.ndarray, z1_data: np.ndarray, control_time: float,
                                     batch_size: int = 1, save_interval: int = 100,
                                     temp_save_dir: str = None, 
                                     return_path: bool = False) -> np.ndarray:
        """
        Extract U-Space representations from UNet bottleneck features
        
        Process:
        1. Interpolate: xt = (1-t) * z0 + t * z1
        2. Forward through UNet encoder to bottleneck
        3. Extract bottleneck features as U-Space
        
        Args:
            z0_data: noise vectors (batch, channels, height, width)
            z1_data: target images (batch, channels, height, width)
            control_time: time t for extraction (0.0 to 1.0)
            batch_size: batch size for processing (reduced for memory safety)
            save_interval: save intermediate results every N batches
            temp_save_dir: directory to save temporary results
            return_path: if True, return file path instead of loading data to memory
            
        Returns:
            uspace_data: U-Space features from UNet bottleneck (or file path if return_path=True)
        """
        logging.info(f"Extracting U-Space at t={control_time} from UNet bottleneck features...")
        logging.info(f"Using batch size: {batch_size}, save interval: {save_interval}")
        
        if self.score_model is None:
            raise ValueError("Model not loaded! Call load_model() first.")
            
        # Convert to tensor if needed
        if isinstance(z0_data, np.ndarray):
            z0_tensor = torch.from_numpy(z0_data)
        else:
            z0_tensor = z0_data
            
        if isinstance(z1_data, np.ndarray):
            z1_tensor = torch.from_numpy(z1_data)
        else:
            z1_tensor = z1_data
            
        num_samples = len(z0_tensor)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        # Create temporary save directory if not provided
        if temp_save_dir is None:
            temp_save_dir = f"temp_uspace_t_{control_time:.2f}"
        os.makedirs(temp_save_dir, exist_ok=True)
        
        all_uspace = []
        temp_file_counter = 0
        
        # Clear GPU cache before starting
        torch.cuda.empty_cache()
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            z0_batch = z0_tensor[start_idx:end_idx].to(self.device)
            z1_batch = z1_tensor[start_idx:end_idx].to(self.device)
            
            try:
                with torch.no_grad():
                    # Step 1: Generate xt from z0 using ODE sampling to control_time (consistent with inference)
                    # Use the same sampling logic as in training/inference
                    xt_batch = self._sample_to_time(z0_batch, control_time)
                    
                    # Step 2: Extract U-Space from UNet bottleneck
                    uspace_batch = self._extract_bottleneck_features(xt_batch, 
                                                                   torch.ones(len(xt_batch), device=self.device) * control_time)
                    
                all_uspace.append(uspace_batch.cpu())
                
                # Force garbage collection after each batch
                del uspace_batch, xt_batch, z0_batch, z1_batch
                torch.cuda.empty_cache()
                gc.collect()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.error(f"GPU out of memory at batch {batch_idx}. Try reducing batch size further.")
                    # Clear cache and try to continue
                    torch.cuda.empty_cache()
                    gc.collect()
                    raise
                else:
                    raise
            
            # Save intermediate results every save_interval batches
            if (batch_idx + 1) % save_interval == 0 or batch_idx == num_batches - 1:
                if all_uspace:  # Only save if we have data
                    # Concatenate current batch
                    temp_tensor = torch.cat(all_uspace, dim=0)
                    temp_data = temp_tensor.cpu().numpy()
                    
                    # Save to temporary file
                    temp_file = os.path.join(temp_save_dir, f"temp_chunk_{temp_file_counter:04d}.npy")
                    np.save(temp_file, temp_data)
                    
                    logging.info(f"Saved temporary chunk {temp_file_counter} ({len(temp_data)} samples) to {temp_file}")
                    
                    # Clear memory
                    del temp_tensor, temp_data, all_uspace
                    all_uspace = []
                    temp_file_counter += 1
                    
                    # Force cleanup
                    torch.cuda.empty_cache()
                    gc.collect()
            
        # Create final result file without loading everything into memory
        logging.info("Creating final result file with streaming concatenation...")
        
        # First pass: get total size and validate files
        total_samples = 0
        valid_files = []
        sample_shape = None
        
        for i in range(temp_file_counter):
            temp_file = os.path.join(temp_save_dir, f"temp_chunk_{i:04d}.npy")
            if os.path.exists(temp_file):
                # Load header only to get shape info
                chunk_data = np.load(temp_file, mmap_mode='r')  # Memory-map mode
                if sample_shape is None:
                    sample_shape = chunk_data.shape[1:]  # Store sample shape
                total_samples += chunk_data.shape[0]
                valid_files.append((temp_file, chunk_data.shape[0]))
                logging.info(f"Found chunk {i} with {chunk_data.shape[0]} samples")
                del chunk_data  # Release memory map
        
        if not valid_files:
            raise RuntimeError("No valid temporary files found")
        
        # Create final output file
        final_shape = (total_samples,) + sample_shape
        logging.info(f"Creating final output file with shape: {final_shape}")
        
        final_result_file = os.path.join(temp_save_dir, "final_uspace_result.npy")
        final_array = np.memmap(final_result_file, dtype=np.float32, mode='w+', shape=final_shape)
        
        # Second pass: copy data chunk by chunk
        current_idx = 0
        for temp_file, chunk_size in valid_files:
            logging.info(f"Copying chunk from {temp_file}...")
            
            # Load chunk data
            chunk_data = np.load(temp_file)
            
            # Copy to final array
            end_idx = current_idx + chunk_size
            final_array[current_idx:end_idx] = chunk_data
            
            current_idx = end_idx
            
            # Clean up immediately
            del chunk_data
            os.remove(temp_file)
            gc.collect()
            
            logging.info(f"Copied chunk, progress: {current_idx}/{total_samples}")
        
        # Flush the memmap to ensure data is written
        del final_array
        
        logging.info(f"Final U-Space result saved to: {final_result_file}")
        
        # Return based on what the caller needs
        if return_path:
            # Return file path for memory efficiency
            return final_result_file
        else:
            # Load the result for backwards compatibility
            uspace_data = np.load(final_result_file)
            # Clean up the file since we loaded it
            os.remove(final_result_file)
            
            # Clean up temporary directory
            try:
                os.rmdir(temp_save_dir)
            except OSError:
                logging.warning(f"Could not remove temporary directory {temp_save_dir}")
            
            logging.info(f"Extracted U-Space data shape: {uspace_data.shape}")
            return uspace_data
        
    def _sample_to_time(self, z0: torch.Tensor, target_time: float) -> torch.Tensor:
        """
        Sample from z0 to target_time using ODE with same steps as config
        
        Args:
            z0: Initial noise tensor
            target_time: Target time (0.0 to 1.0)
            
        Returns:
            xt: Tensor at target_time using ODE steps
        """
        if self.score_model is None:
            raise ValueError("Model not loaded!")
            
        # Get sample_N from config (default 10 for fast sampling)
        N = getattr(self.config.sampling, 'sample_N', 10)
        eps = 1e-3
        
        # Use euler sampling for consistency and speed
        dt = target_time / N
        x = z0.detach().clone()
        
        model_fn = mutils.get_model_fn(self.score_model, train=False)
        shape = z0.shape
        device = z0.device
        
        # ODE steps from 0 to target_time
        for i in range(N):
            if target_time <= eps:
                break
                
            num_t = i / N * (target_time - eps) + eps
            if num_t >= target_time:
                break
                
            t = torch.ones(shape[0], device=device) * num_t
            
            # Get model prediction
            pred = model_fn(x, t * 999)  # Scale time for model
            
            # Euler step
            x = x.detach().clone() + pred * dt
            
        return x
        
    def _extract_bottleneck_features(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Extract bottleneck features from UNet using forward hooks
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            t: Time tensor (batch_size,)
            
        Returns:
            bottleneck_features: UNet bottleneck features as U-Space
        """
        # Register forward hook to capture bottleneck features
        bottleneck_features = []
        
        def hook_fn(module, input, output):
            # Capture the bottleneck features
            bottleneck_features.append(output.clone())
        
        # Find the bottleneck layer (middle of the UNet)
        # For NCSN++ architecture, look for the middle block
        hook_handle = None
        target_module = None
        
        # Try to find the middle/bottleneck layer
        if hasattr(self.score_model, 'all_modules'):
            # For NCSN++ models with all_modules
            modules = list(self.score_model.all_modules)
            middle_idx = len(modules) // 2
            target_module = modules[middle_idx]
        elif hasattr(self.score_model, 'module_list') or hasattr(self.score_model, 'modules'):
            # Alternative module access
            modules = list(self.score_model.modules())
            # Find conv layers and take one from the middle
            conv_modules = [m for m in modules if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))]
            if conv_modules:
                middle_idx = len(conv_modules) // 2
                target_module = conv_modules[middle_idx]
        
        if target_module is None:
            # Fallback: use the entire model output and spatially pool
            logging.warning("Could not find bottleneck layer, using model output with pooling")
            with torch.no_grad():
                model_output = self.score_model(x, t)
                # Global average pooling to create bottleneck-like features
                bottleneck = torch.mean(model_output, dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
                return bottleneck
        
        # Register hook
        hook_handle = target_module.register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                # Forward pass to trigger the hook
                _ = self.score_model(x, t)
                
                if bottleneck_features:
                    # Use the captured features
                    features = bottleneck_features[0]
                    
                    # If features are spatial (H, W > 1), pool them to create compact representation
                    if len(features.shape) == 4 and features.shape[-1] > 1:
                        # Global average pooling
                        pooled_features = torch.mean(features, dim=[2, 3], keepdim=True)
                        return pooled_features
                    else:
                        return features
                else:
                    logging.warning("No bottleneck features captured, using fallback")
                    # Fallback: use spatial pooling of input
                    return torch.mean(x, dim=[2, 3], keepdim=True)
                    
        finally:
            # Clean up hook
            if hook_handle is not None:
                hook_handle.remove()
        
    def _extract_uspace_all_times(self, z0_data: np.ndarray, z1_data: np.ndarray, 
                                  time_points: List[float]) -> List[np.ndarray]:
        """
        Extract U-Space representations for all time points simultaneously
        
        Args:
            z0_data: noise vectors (batch, channels, height, width)
            z1_data: target images (batch, channels, height, width)
            time_points: list of time points to extract
            
        Returns:
            List of U-Space data arrays, one for each time point
        """
        logging.info(f"Extracting U-Space for {len(time_points)} time points simultaneously...")
        
        if self.score_model is None:
            raise ValueError("Model not loaded! Call load_model() first.")
            
        # Convert to tensor if needed
        if isinstance(z0_data, np.ndarray):
            z0_tensor = torch.from_numpy(z0_data)
        else:
            z0_tensor = z0_data
            
        if isinstance(z1_data, np.ndarray):
            z1_tensor = torch.from_numpy(z1_data)
        else:
            z1_tensor = z1_data
            
        num_samples = len(z0_tensor)
        batch_size = 1  # Process one sample at a time for memory safety
        
        # Initialize results for all time points
        all_time_results = [[] for _ in time_points]
        
        # Clear GPU cache before starting
        torch.cuda.empty_cache()
        
        for sample_idx in range(num_samples):
            z0_sample = z0_tensor[sample_idx:sample_idx+1].to(self.device)
            z1_sample = z1_tensor[sample_idx:sample_idx+1].to(self.device)
            
            try:
                with torch.no_grad():
                    # Process all time points for this sample
                    sample_results = []
                    
                    for t in time_points:
                        # Generate xt using ODE sampling
                        xt_sample = self._sample_to_time(z0_sample, t)
                        
                        # Extract U-Space from UNet bottleneck
                        uspace_sample = self._extract_bottleneck_features(
                            xt_sample, 
                            torch.ones(1, device=self.device) * t
                        )
                        
                        sample_results.append(uspace_sample.cpu())
                    
                    # Store results for each time point
                    for i, result in enumerate(sample_results):
                        all_time_results[i].append(result)
                    
                # Clean up immediately
                del z0_sample, z1_sample, sample_results
                torch.cuda.empty_cache()
                gc.collect()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.error(f"GPU out of memory at sample {sample_idx}. Try reducing batch size further.")
                    torch.cuda.empty_cache()
                    gc.collect()
                    raise
                else:
                    raise
        
        # Concatenate results for each time point
        final_results = []
        for i, time_result_list in enumerate(all_time_results):
            if time_result_list:
                concatenated = torch.cat(time_result_list, dim=0)
                final_results.append(concatenated.cpu().numpy())
            else:
                final_results.append(np.array([]))
        
        logging.info(f"Extracted U-Space for {len(time_points)} time points, shapes: {[r.shape for r in final_results]}")
        return final_results

    def process_existing_data_pairs_efficient(self, data_path: str, output_dir: str,
                                   time_points: List[float] = None,
                                   method: str = 'true_uspace',
                                   checkpoint_path: str = None,
                                   iteration: int = None):
        """
        Process existing (z0, z1) pairs to extract TRUE U-Space representations with memory-efficient batch processing
        
        Args:
            data_path: path to existing data pairs file (.pt or .pkl) or directory
            output_dir: directory to save extracted u-space data
            time_points: list of time points to extract
            method: extraction method (only 'true_uspace' supported)
            checkpoint_path: path to model checkpoint (required for true_uspace)
            iteration: iteration number for organizing outputs
        """
        logging.info(f"Processing existing data pairs from: {data_path} (memory-efficient)")
        
        # If iteration is specified, create iteration-specific subdirectory
        if iteration is not None:
            output_dir = os.path.join(output_dir, f'iteration_{iteration}', 'uspace_extracted')
        
        # Default time points if not specified
        if time_points is None:
            time_points = [0.1, 0.25, 0.5, 0.75]
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model if using true_uspace method
        if method == 'true_uspace':
            if checkpoint_path is None:
                raise ValueError("checkpoint_path required for true_uspace method")
            self.load_model(checkpoint_path)
        
        # Memory-efficient data processing
        if os.path.isdir(data_path):
            # Handle directory with multiple files - process in batches
            summary_file = self._process_directory_batch(data_path, output_dir, time_points, method)
        else:
            # Handle single file
            summary_file = self._process_single_file(data_path, output_dir, time_points, method)
            
        logging.info(f"Saved TRUE U-Space extraction summary to: {summary_file}")
        logging.info("TRUE U-Space extraction completed with memory-efficient processing!")
        
        return summary_file
    
    def _process_directory_batch(self, data_path: str, output_dir: str, time_points: List[float], method: str):
        """Process directory with multiple data files in memory-efficient batches"""
        logging.info(f"Processing directory in batches: {data_path}")
        
        # Find all data files
        data_files = []
        for file in os.listdir(data_path):
            if file.endswith('.pt') or file.endswith('.pkl'):
                data_files.append(os.path.join(data_path, file))
        
        if not data_files:
            raise ValueError(f"No .pt or .pkl files found in directory: {data_path}")
        
        logging.info(f"Found {len(data_files)} data files to process")
        
        # Process files in small batches to save memory
        batch_size = 8  # Process 8 files at a time
        total_samples = 0
        uspace_results = {t: [] for t in time_points}  # Store file paths for each time point
        
        num_batches = (len(data_files) + batch_size - 1) // batch_size
        batch_progress = tqdm(range(0, len(data_files), batch_size), desc="Processing data batches for U-Space extraction")
        
        for batch_start in batch_progress:
            batch_end = min(batch_start + batch_size, len(data_files))
            batch_files = data_files[batch_start:batch_end]
            
            # Update progress bar description with current batch info
            batch_num = batch_start // batch_size + 1
            batch_progress.set_description(f"Processing batch {batch_num}/{num_batches}")
            
            # Collect data from this batch
            z0_batch_list = []
            z1_batch_list = []
            
            for data_file in batch_files:
                try:
                    if data_file.endswith('.pt'):
                        data = torch.load(data_file, map_location='cpu')
                    else:
                        with open(data_file, 'rb') as f:
                            data = pickle.load(f)
                    
                    # Handle different data formats
                    if isinstance(data, dict):
                        if 'z0' in data and 'z1' in data:
                            # Format: {'z0': tensor, 'z1': tensor}
                            z0_batch_list.append(data['z0'])
                            z1_batch_list.append(data['z1'])
                        elif 'z0_batch' in data and 'z1_batch' in data:
                            # Format: {'z0_batch': tensor, 'z1_batch': tensor, ...}
                            z0_batch_list.append(data['z0_batch'])
                            z1_batch_list.append(data['z1_batch'])
                        else:
                            logging.warning(f"Skipping file {data_file}: unknown dict format with keys {list(data.keys())}")
                            continue
                    else:
                        # Handle files that are just tensors (like images_batch_*.pt)
                        filename = os.path.basename(data_file)
                        
                        # Skip probability files and other non-data files
                        if any(skip_pattern in filename for skip_pattern in ['probabilities', 'labels', 'metadata', 'weights']):
                            logging.debug(f"Skipping non-data file: {data_file}")
                            continue
                        
                        # Handle image files
                        if 'images_batch' in filename or 'image_batch' in filename:
                            # This is z1 (target images)
                            z1_batch_list.append(data)
                            # Create corresponding noise as z0
                            noise = torch.randn_like(data)
                            z0_batch_list.append(noise)
                        elif 'noise_batch' in filename:
                            # This is z0 (noise), need to find corresponding images
                            # For now, skip single noise files
                            logging.debug(f"Skipping noise-only file: {data_file}")
                            continue
                        else:
                            # Check if it's a valid tensor with image-like dimensions
                            if isinstance(data, torch.Tensor) and len(data.shape) == 4:
                                # Assume it's image data and treat as z1
                                logging.info(f"Treating tensor file {data_file} as image data (z1)")
                                z1_batch_list.append(data)
                                # Create corresponding noise as z0
                                noise = torch.randn_like(data)
                                z0_batch_list.append(noise)
                            else:
                                logging.debug(f"Skipping file {data_file}: not image-like tensor (shape: {data.shape if hasattr(data, 'shape') else 'unknown'})")
                                continue
                            
                except Exception as e:
                    logging.warning(f"Error loading {data_file}: {e}")
                    continue
            
            if not z0_batch_list or not z1_batch_list:
                logging.warning(f"No valid data found in batch {batch_start//batch_size + 1}")
                continue
            
            # Concatenate batch data
            z0_batch = torch.cat(z0_batch_list, dim=0)
            z1_batch = torch.cat(z1_batch_list, dim=0)
            
            logging.info(f"Batch contains {len(z0_batch)} samples")
            total_samples += len(z0_batch)
            
            # Process this batch for all time points together
            logging.info(f"Processing batch for all time points: {time_points}")
            
            # Extract U-Space for all time points in one go
            batch_uspace_results = self._extract_uspace_all_times(
                z0_batch.numpy(), z1_batch.numpy(), time_points
            )
            
            # Save results for each time point
            for i, t in enumerate(time_points):
                final_file = os.path.join(output_dir, f'batch_{batch_start//batch_size + 1}_uspace_t_{t:.2f}.npy')
                np.save(final_file, batch_uspace_results[i])
                uspace_results[t].append(final_file)
            
            # Clear memory
            del z0_batch, z1_batch, z0_batch_list, z1_batch_list
            torch.cuda.empty_cache()
            gc.collect()
        
        # Combine all batch results for each time point
        logging.info(f"Combining batch results for {len(time_points)} time points...")
        for t in time_points:
            if uspace_results[t]:
                combined_file = os.path.join(output_dir, f'uspace_t_{t:.2f}.npy')
                self._combine_batch_files(uspace_results[t], combined_file)
                uspace_results[t] = combined_file
        
        # Create summary
        summary_data = {
            'total_samples': total_samples,
            'num_batches_processed': (len(data_files) + batch_size - 1) // batch_size,
            'time_points': time_points,
            'method': method,
            'file_paths': uspace_results,
            'metadata': {
                'num_samples': total_samples,
                'time_points': time_points,
                'extraction_method': method,
                'batch_processing': True,
                'is_true_uspace': True,
                'bottleneck_features': True
            }
        }
        
        summary_file = os.path.join(output_dir, 'true_uspace_extraction_summary.pkl')
        with open(summary_file, 'wb') as f:
            pickle.dump(summary_data, f)
            
        return summary_file
    
    def _process_single_file(self, data_path: str, output_dir: str, time_points: List[float], method: str):
        """Process single data file"""
        logging.info(f"Processing single file: {data_path}")
        
        # Load data
        if data_path.endswith('.pt'):
            data_dict = torch.load(data_path, map_location='cpu')
        else:
            with open(data_path, 'rb') as f:
                data_dict = pickle.load(f)
        
        if not isinstance(data_dict, dict) or 'z0' not in data_dict or 'z1' not in data_dict:
            raise ValueError(f"Single file must contain dict with 'z0' and 'z1' keys")
            
        z0_data = data_dict['z0']
        z1_data = data_dict['z1']
        
        logging.info(f"Loaded z0 shape: {z0_data.shape}")
        logging.info(f"Loaded z1 shape: {z1_data.shape}")
        
        # Extract U-Space representations for all time points simultaneously
        logging.info(f"Extracting U-Space for all time points: {time_points}")
        uspace_data_list = self._extract_uspace_all_times(
            z0_data.numpy() if hasattr(z0_data, 'numpy') else z0_data,
            z1_data.numpy() if hasattr(z1_data, 'numpy') else z1_data,
            time_points
        )
        
        # Save results for each time point
        uspace_results = {}
        for i, t in enumerate(time_points):
            output_file = os.path.join(output_dir, f'uspace_t_{t:.2f}.npy')
            np.save(output_file, uspace_data_list[i])
            uspace_results[t] = output_file
            
            # Save metadata
            metadata_file = os.path.join(output_dir, f'uspace_t_{t:.2f}_metadata.pkl')
            with open(metadata_file, 'wb') as f:
                pickle.dump({
                    'time': t,
                    'method': method,
                    'shape': uspace_data_list[i].shape,
                    'extraction_time': t,
                    'method_used': method,
                    'data_file': output_file,
                    'is_true_uspace': True,
                    'bottleneck_features': True
                }, f)
                
            logging.info(f"Saved TRUE U-Space data for t={t} to: {output_file}")
            
        # Clear data from memory
        del uspace_data_list
        gc.collect()
        
        # Create summary
        summary_data = {
            'z0_shape': z0_data.shape,
            'z1_shape': z1_data.shape,
            'time_points': time_points,
            'method': method,
            'file_paths': uspace_results,
            'metadata': {
                'num_samples': len(z0_data),
                'time_points': time_points,
                'extraction_method': method,
                'incremental_save': True,
                'is_true_uspace': True,
                'bottleneck_features': True
            }
        }
        
        summary_file = os.path.join(output_dir, 'true_uspace_extraction_summary.pkl')
        with open(summary_file, 'wb') as f:
            pickle.dump(summary_data, f)
            
        return summary_file
    
    def _combine_batch_files(self, batch_files: List[str], output_file: str):
        """Combine multiple batch result files into one file using memory mapping"""
        # First pass: get total size
        total_samples = 0
        sample_shape = None
        
        for batch_file in batch_files:
            data = np.load(batch_file, mmap_mode='r')
            if sample_shape is None:
                sample_shape = data.shape[1:]
            total_samples += data.shape[0]
            del data
        
        # Create output file
        final_shape = (total_samples,) + sample_shape
        final_array = np.memmap(output_file, dtype=np.float32, mode='w+', shape=final_shape)
        
        # Second pass: copy data
        current_idx = 0
        for batch_file in batch_files:
            data = np.load(batch_file)
            end_idx = current_idx + data.shape[0]
            final_array[current_idx:end_idx] = data
            current_idx = end_idx
            
            # Clean up batch file
            del data
            os.remove(batch_file)
        
        del final_array
        logging.info(f"Combined {len(batch_files)} batch files into {output_file}")


def main():
    """Main function to extract TRUE U-Space from existing data pairs"""
    import argparse
    from configs.celeba_256_semantic_reflow_iterative import get_config
    
    parser = argparse.ArgumentParser(description='Extract TRUE U-Space from Flow Matching')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to existing (z0, z1) data pairs file')
    parser.add_argument('--output_dir', type=str, default='workdir/true_uspace_extracted',
                       help='Output directory for TRUE U-Space data')
    parser.add_argument('--time_points', nargs='+', type=float, 
                       default=[0.1, 0.25, 0.5, 0.75],
                       help='Time points to extract (e.g., 0.1 0.25 0.5)')
    parser.add_argument('--method', type=str, choices=['true_uspace'],
                       default='true_uspace',
                       help='Extraction method (only true_uspace supported - extracts UNet bottleneck features)')
    parser.add_argument('--checkpoint_path', type=str,
                       help='Path to model checkpoint (required for true_uspace method)')
    parser.add_argument('--control_time', type=float, default=0.25,
                       help='Main control time point')
    parser.add_argument('--iteration', type=int,
                       help='Iteration number for organizing outputs in iteration subdirectories')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Get configuration
    config = get_config()
    
    # Validate arguments
    if args.method == 'true_uspace' and args.checkpoint_path is None:
        raise ValueError("--checkpoint_path is required when using true_uspace method")
        
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
        
    # Create TRUE U-Space extractor
    extractor = USpaceExtractor(config, control_time=args.control_time)
    
    # Process existing data using memory-efficient method
    combined_file = extractor.process_existing_data_pairs_efficient(
        data_path=args.data_path,
        output_dir=args.output_dir,
        time_points=args.time_points,
        method=args.method,
        checkpoint_path=args.checkpoint_path,
        iteration=args.iteration
    )
    
    logging.info(f"TRUE U-Space extraction completed!")
    logging.info(f"Combined data saved to: {combined_file}")
    logging.info(f"Individual time point files saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
