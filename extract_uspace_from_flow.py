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

        
    def _sample_to_time(self, z0: torch.Tensor, target_time: float) -> torch.Tensor:
        """Sample from z0 to target_time using Euler method"""
        if self.score_model is None:
            raise ValueError("Model not loaded!")
        
        # Use the same approach as the reference sampling code
        with torch.no_grad():
            x = z0.detach().clone()
            
            # Check inputs
            if torch.isnan(x).any():
                print(f"Input z0 has NaN!")
                return torch.zeros_like(x)
            
            # Get model function (same as reference)
            model_fn = mutils.get_model_fn(self.score_model, train=False)
            
            # Calculate number of steps based on target_time
            # Use similar discretization as reference code
            eps = 1e-3
            N_steps = max(1, int(target_time * 100))  # Scale steps with target_time
            dt = (target_time - eps) / N_steps
            
            #print(f"Sampling to t={target_time}, N_steps={N_steps}, dt={dt}")
            
            current_time = eps
            for i in range(N_steps):
                # Current time (same scaling as reference: t*999)
                t = torch.ones(x.shape[0], device=x.device) * current_time
                
                # Get prediction
                pred = model_fn(x, t * 999)
                
                # Debug: check for NaN
                if torch.isnan(pred).any():
                    print(f"Model prediction has NaN at t={current_time}")
                    print(f"Input x stats: min={x.min():.4f}, max={x.max():.4f}")
                    print(f"Input x has NaN: {torch.isnan(x).any()}")
                    return torch.zeros_like(x)
                
                # Apply sigma correction (from reference code)
                if hasattr(self, 'sde') and hasattr(self.sde, 'sigma_t'):
                    sigma_t = self.sde.sigma_t(current_time)
                    noise_scale = getattr(self.sde, 'noise_scale', 1.0)
                    
                    pred_sigma = pred + (sigma_t**2)/(2*(noise_scale**2)*((1.-current_time)**2)) * \
                                (0.5 * current_time * (1.-current_time) * pred - 0.5 * (2.-current_time) * x.detach().clone())
                    
                    # Euler step with noise
                    x = x.detach().clone() + pred_sigma * dt + \
                        sigma_t * np.sqrt(dt) * torch.randn_like(pred_sigma).to(x.device)
                else:
                    # Simple Euler step (deterministic)
                    x = x.detach().clone() + pred * dt
                
                current_time += dt
                
                # Check for NaN after update
                if torch.isnan(x).any():
                    print(f"x becomes NaN at step {i}, t={current_time}")
                    return torch.zeros_like(x)
            
            return x

        
    def _extract_bottleneck_features(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Extract bottleneck features from NCSNpp using forward hooks"""
        bottleneck_features = []
        #print(f"Input x stats: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}, std={x.std():.4f}")
        #print(f"Input x has NaN: {torch.isnan(x).any()}")
        #print(f"Input t: {t}")
        #print(f"Input t has NaN: {torch.isnan(t).any()}")
        
        def hook_fn(module, input, output):
            bottleneck_features.append(output.clone())
        
        # Get model modules
        try:
            all_modules = self.score_model.module.all_modules if hasattr(self.score_model, 'module') else self.score_model.all_modules
            
            # Find attention layers (ideal bottleneck)
            attention_indices = [i for i, m in enumerate(all_modules) 
                            if 'AttnBlockpp' in str(type(m))]
            
            if attention_indices:
                target_module = all_modules[attention_indices[0]]  # Use first attention layer
            else:
                # Fallback: use middle of bottleneck range (modules 20-34)
                target_module = all_modules[27]
                
        except:
            # Final fallback: use model output
            with torch.no_grad():
                model_output = self.score_model(x, t)
                if len(model_output.shape) == 4:
                    return torch.mean(model_output, dim=[2, 3], keepdim=True)
                return model_output
        
        # Register hook and forward pass
        hook_handle = target_module.register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                _ = self.score_model(x, t)
                #print(bottleneck_features)
                if bottleneck_features:
                    features = bottleneck_features[0]
                    # Pool spatial dimensions if needed
                    if len(features.shape) == 4 and (features.shape[2] > 1 or features.shape[3] > 1):
                        return torch.mean(features, dim=[2, 3], keepdim=True)
                    return features
                else:
                    # Hook failed fallback
                    model_output = self.score_model(x, t)
                    if len(model_output.shape) == 4:
                        return torch.mean(model_output, dim=[2, 3], keepdim=True)
                    return model_output
                    
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        
    def extract_uspace(self, z0_data: np.ndarray, 
                                  time_points: List[float]) -> List[np.ndarray]:
        """
        Extract U-Space representations for all time points simultaneously
        
        Args:
            z0_data: noise vectors (batch, channels, height, width)
            time_points: list of time points to extract
            
        Returns:
            List of U-Space data arrays, one for each time point
        """
        #logging.info(f"Extracting U-Space for {len(time_points)} time points simultaneously...")
        #print(f"Extracting U-Space for {len(time_points)} time points simultaneously...")
        
        if self.score_model is None:
            raise ValueError("Model not loaded! Call load_model() first.")
            
        # Convert to tensor if needed
        if isinstance(z0_data, np.ndarray):
            z0_tensor = torch.from_numpy(z0_data)
        else:
            z0_tensor = z0_data
            
            
        num_samples = len(z0_tensor)
        batch_size = 1  # Process one sample at a time for memory safety
        
        # Initialize results for all time points
        all_time_results = [[] for _ in time_points]
        
        # Clear GPU cache before starting
        torch.cuda.empty_cache()
        
        for sample_idx in range(num_samples):
            z0_sample = z0_tensor[sample_idx:sample_idx+1].to(self.device)
            
            try:
                with torch.no_grad():
                    # Process all time points for this sample
                    sample_results = []
                    
                    for t in time_points:
                        # Generate xt using ODE sampling
                        xt_sample = self._sample_to_time(z0_sample, t)
                        #print(torch.isnan(z0_sample).any())
                        #print(torch.isnan(xt_sample).any())
                        # Extract U-Space from UNet bottleneck
                        uspace_sample = self._extract_bottleneck_features(
                            xt_sample, 
                            torch.ones(1, device=self.device) * t
                        )
                        if torch.isnan(uspace_sample).any():
                            logging.warning(f"NaN detected in U-Space at t={t}, sample {sample_idx}")
                            uspace_sample = torch.zeros_like(uspace_sample)
                        
                        sample_results.append(uspace_sample.cpu())
                    
                    # Store results for each time point
                    for i, result in enumerate(sample_results):
                        all_time_results[i].append(result)
                    
                # Clean up immediately
                del z0_sample, sample_results
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
                final_results.append(concatenated.cpu().numpy().astype(np.float32))
            else:
                final_results.append(np.array([], dtype=np.float32))
        
        #logging.info(f"Extracted U-Space for {len(time_points)} time points, shapes: {[r.shape for r in final_results]}")
        return final_results

    def process(self, data_path: str, output_dir: str,
                                   time_points: List[float] = None,
                                   checkpoint_path: str = None,
                                   iteration: int = None):
        logging.info(f"Processing existing data pairs from: {data_path} (memory-efficient)")
        
        if iteration is not None:
            base_output_dir = os.path.join(output_dir, f'iteration_{iteration}', 'uspace_extracted')
        else:
            base_output_dir = output_dir
        if time_points is None:
            time_points = [0.1, 0.2, 0.3, 0.5]
        if checkpoint_path is None:
            raise ValueError("checkpoint_path required")
        self.load_model(checkpoint_path)
        
        if os.path.isdir(data_path):
            summary_file = self.process_batch(data_path, base_output_dir, time_points)
        else:
            print("Data path is not a directory")
            
        logging.info(f"Saved U-Space extraction summary to: {summary_file}")
        logging.info("U-Space extraction completed with memory-efficient processing!")
        
        return summary_file
    
    def process_batch(self, data_path: str, output_dir: str, time_points: List[float]):
        logging.info(f"Processing directory in batches: {data_path}")
        
        # Find all data files
        data_files = []
        for file in os.listdir(data_path):
            if file.startswith('noise') and file.endswith('.pt'):
                data_files.append(os.path.join(data_path, file))
        
        if not data_files:
            raise ValueError(f"No .pt or .pkl files found in directory: {data_path}")
        
        # Sort files by batch number to match label order
        data_files.sort(key=lambda x: int(os.path.basename(x).split('_batch_')[1].split('.pt')[0]))
        
        logging.info(f"Found {len(data_files)} data files to process")
        
        batch_size = 2
        total_samples = 0
        uspace_results = {t: [] for t in time_points} 
        
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
            
            for data_file in batch_files:
                try:
                    if data_file.endswith('.pt'):
                        data = torch.load(data_file, map_location='cpu')
                    if isinstance(data, dict) and 'z0' in data:
                        z0_batch_list.append(data['z0'])
                    elif isinstance(data, torch.Tensor):
                        z0_batch_list.append(data)
                    else:
                        logging.warning(f"Skipping file {data_file}: not dict with 'z0' or tensor")
                        continue
                except Exception as e:
                    logging.warning(f"Error loading {data_file}: {e}")
                    continue
            
            if not z0_batch_list:
                logging.warning(f"No valid z0 data found in batch {batch_start//batch_size + 1}")
                continue
            
            z0_batch = torch.cat(z0_batch_list, dim=0)
            total_samples += len(z0_batch)
            
            # Process this batch for all time points together

            #logging.info(f"Batch contains {len(z0_batch)} samples")
            total_samples += len(z0_batch)
            
            # Process this batch for all time points together
            #logging.info(f"Processing batch for all time points: {time_points}")
            
            # Extract U-Space for all time points in one go
            batch_uspace_results = self.extract_uspace(
                z0_batch.numpy(), time_points
            )
            if batch_num == 1:
                tqdm.write(f"Batch {batch_num} contains {len(z0_batch)} samples")
                tqdm.write(f"Processing batch {batch_num} for time points: {time_points}")
                tqdm.write(f"Batch {batch_num} - Extracted U-Space shapes: {[r.shape for r in batch_uspace_results]}")


            # Save results for each time point in separate folders
            for i, t in enumerate(time_points):
                t_output_dir = os.path.join(output_dir, f't_{t:.2f}')
                os.makedirs(t_output_dir, exist_ok=True)
                final_file = os.path.join(t_output_dir, f'batch_{batch_start//batch_size + 1}_uspace_t_{t:.2f}.npy')
                np.save(final_file, batch_uspace_results[i])
                uspace_results[t].append(final_file)
            
            # Clear memory
            del z0_batch, z0_batch_list
            torch.cuda.empty_cache()
            gc.collect()
        
        # Combine all batch results for each time point
        logging.info(f"Combining batch results for {len(time_points)} time points...")
        for t in time_points:
            if uspace_results[t]:
                t_output_dir = os.path.join(output_dir, f't_{t:.2f}')
                combined_file = os.path.join(t_output_dir, f'uspace_t_{t:.2f}.npy')
                self._combine_batch_files(uspace_results[t], combined_file)
                uspace_results[t] = combined_file
        
        # Create summary
        summary_data = {
            'total_samples': total_samples,
            'num_batches_processed': (len(data_files) + batch_size - 1) // batch_size,
            'time_points': time_points,
            'file_paths': uspace_results,
            'metadata': {
                'num_samples': total_samples,
                'time_points': time_points,
                'batch_processing': True,
                'is_true_uspace': True,
                'bottleneck_features': True
            }
        }
        
        summary_file = os.path.join(output_dir, 'true_uspace_extraction_summary.pkl')
        with open(summary_file, 'wb') as f:
            pickle.dump(summary_data, f)
            
        return summary_file
    
    def _combine_batch_files(self, batch_files: List[str], output_file: str):
        """Combine multiple batch result files into one file"""
        all_data = []
        
        for batch_file in batch_files:
            data = np.load(batch_file)
            all_data.append(data)
        
        if not all_data:
            raise ValueError("No valid batch files to combine")
        
        combined_data = np.concatenate(all_data, axis=0)
        np.save(output_file, combined_data)
        
        # Clean up batch files
        for batch_file in batch_files:
            os.remove(batch_file)
        
        logging.info(f"Combined {len(batch_files)} files into {output_file}, shape: {combined_data.shape}")



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
    parser.add_argument('--checkpoint_path', type=str,
                       help='Path to model checkpoint')
    parser.add_argument('--control_time', type=float, default=0.5,
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
    if args.checkpoint_path is None:
        raise ValueError("--checkpoint_path is required")
        
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
        
    extractor = USpaceExtractor(config, control_time=args.control_time)
    
    combined_file = extractor.process(
        data_path=args.data_path,
        output_dir=args.output_dir,
        time_points=getattr(args, 'time_points', None),
        checkpoint_path=getattr(args, 'checkpoint_path', None),
        iteration=getattr(args, 'iteration', None)
    )
    
    logging.info(f"TRUE U-Space extraction completed!")
    logging.info(f"Combined data saved to: {combined_file}")
    logging.info(f"Individual time point files saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
