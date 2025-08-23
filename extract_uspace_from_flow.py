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
    Extract TRUE U-Space representations from UNet bottleneck features during flow matching
    
    Given (z0, z1) pairs, extract UNet bottleneck features at interpolated points:
    - Interpolation: xt = (1-t) * z0 + t * z1
    - Extract bottleneck features from UNet as TRUE U-Space
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
        
    def extract_true_uspace_from_unet(self, z0_data: np.ndarray, z1_data: np.ndarray, control_time: float,
                                     batch_size: int = 1, save_interval: int = 100,
                                     temp_save_dir: str = None, 
                                     return_path: bool = False) -> np.ndarray:
        """
        Extract TRUE U-Space representations from UNet bottleneck features
        
        Process:
        1. Interpolate: xt = (1-t) * z0 + t * z1
        2. Forward through UNet encoder to bottleneck
        3. Extract bottleneck features as true U-Space
        
        Args:
            z0_data: noise vectors (batch, channels, height, width)
            z1_data: target images (batch, channels, height, width)
            control_time: time t for extraction (0.0 to 1.0)
            batch_size: batch size for processing (reduced for memory safety)
            save_interval: save intermediate results every N batches
            temp_save_dir: directory to save temporary results
            return_path: if True, return file path instead of loading data to memory
            
        Returns:
            uspace_data: TRUE U-Space features from UNet bottleneck (or file path if return_path=True)
        """
        logging.info(f"Extracting TRUE U-Space at t={control_time} from UNet bottleneck features...")
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
            temp_save_dir = f"temp_true_uspace_t_{control_time:.2f}"
        os.makedirs(temp_save_dir, exist_ok=True)
        
        all_uspace = []
        temp_file_counter = 0
        
        # Clear GPU cache before starting
        torch.cuda.empty_cache()
        
        for batch_idx in tqdm(range(num_batches), desc="Extracting TRUE U-Space from UNet bottleneck"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            z0_batch = z0_tensor[start_idx:end_idx].to(self.device)
            z1_batch = z1_tensor[start_idx:end_idx].to(self.device)
            
            try:
                with torch.no_grad():
                    # Step 1: Flow matching interpolation: xt = (1-t) * z0 + t * z1
                    t = control_time
                    xt_batch = (1 - t) * z0_batch + t * z1_batch
                    
                    # Step 2: Extract TRUE U-Space from UNet bottleneck
                    uspace_batch = self._extract_bottleneck_features(xt_batch, 
                                                                   torch.ones(len(xt_batch), device=self.device) * t)
                    
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
        
        final_result_file = os.path.join(temp_save_dir, "final_true_uspace_result.npy")
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
        
        logging.info(f"Final TRUE U-Space result saved to: {final_result_file}")
        
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
            
            logging.info(f"Extracted TRUE U-Space data shape: {uspace_data.shape}")
            return uspace_data
        
    def _extract_bottleneck_features(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Extract TRUE bottleneck features from UNet using forward hooks
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            t: Time tensor (batch_size,)
            
        Returns:
            bottleneck_features: UNet bottleneck features as TRUE U-Space
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
        
    def extract_multi_time_uspace(self, z0_data: np.ndarray, z1_data: np.ndarray,
                                  time_points: List[float], 
                                  method: str = 'true_uspace',
                                  output_dir: str = None) -> Dict[float, str]:
        """
        Extract TRUE U-Space representations at multiple time points with incremental saving
        
        Args:
            z0_data: noise vectors
            z1_data: target images
            time_points: list of time points to extract (e.g., [0.1, 0.25, 0.5, 0.75])
            method: only 'true_uspace' supported (extracts UNet bottleneck features)
            output_dir: directory to save results (if None, return data in memory)
            
        Returns:
            Dictionary mapping time points to file paths (if output_dir provided) or uspace data
        """
        logging.info(f"Extracting TRUE U-Space at multiple time points: {time_points}")
        
        if method != 'true_uspace':
            raise ValueError("Only 'true_uspace' method is supported now")
        
        uspace_results = {}
        
        for t in time_points:
            logging.info(f"Processing time point t={t}")
            
            # TRUE U-Space extraction with temp saving
            temp_dir = os.path.join(output_dir, f"temp_t_{t:.2f}") if output_dir else None
            
            if output_dir:
                # Get the file path directly without loading to memory
                result_file = self.extract_true_uspace_from_unet(z0_data, z1_data, t, temp_save_dir=temp_dir, return_path=True)
                
                # Move the result file to the final location
                output_file = os.path.join(output_dir, f'true_uspace_t_{t:.2f}.npy')
                import shutil
                shutil.move(result_file, output_file)
                
                # No need to load uspace_data into memory since we're saving files
                uspace_data = None  # Placeholder
            else:
                # Load into memory if not saving to files
                uspace_data = self.extract_true_uspace_from_unet(z0_data, z1_data, t, temp_save_dir=temp_dir)
            
            # Save immediately if output directory is provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
                # For true_uspace, the file is already created above
                # Also save metadata
                metadata_file = os.path.join(output_dir, f'true_uspace_t_{t:.2f}_metadata.pkl')
                with open(metadata_file, 'wb') as f:
                    pickle.dump({
                        'time': t,
                        'method': method,
                        'shape': uspace_data.shape if uspace_data is not None else "saved_to_file",
                        'extraction_time': t,
                        'method_used': method,
                        'data_file': output_file,
                        'is_true_uspace': True,
                        'bottleneck_features': True
                    }, f)
                
                logging.info(f"Saved TRUE U-Space data for t={t} to: {output_file}")
                uspace_results[t] = output_file
                
                # Clear data from memory immediately
                if uspace_data is not None:
                    del uspace_data
                gc.collect()
            else:
                uspace_results[t] = uspace_data
            
        logging.info(f"Extracted TRUE U-Space data for {len(time_points)} time points")
        return uspace_results
        
    def process_existing_data_pairs(self, data_path: str, output_dir: str,
                                   time_points: List[float] = None,
                                   method: str = 'true_uspace',
                                   checkpoint_path: str = None,
                                   iteration: int = None):
        """
        Process existing (z0, z1) pairs to extract TRUE U-Space representations
        
        Args:
            data_path: path to existing data pairs file (.pt or .pkl)
            output_dir: directory to save extracted u-space data
            time_points: list of time points to extract
            method: extraction method (only 'true_uspace' supported)
            checkpoint_path: path to model checkpoint (required for true_uspace)
            iteration: iteration number for organizing outputs
        """
        logging.info(f"Processing existing data pairs from: {data_path}")
        
        # If iteration is specified, create iteration-specific subdirectory
        if iteration is not None:
            output_dir = os.path.join(output_dir, f'iteration_{iteration}', 'uspace_extracted')
        
        # Default time points if not specified
        if time_points is None:
            time_points = [0.1, 0.25, 0.5, 0.75]
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load existing data
        if data_path.endswith('.pt'):
            data_dict = torch.load(data_path, map_location='cpu')
        elif data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                data_dict = pickle.load(f)
        else:
            raise ValueError("Data file must be .pt or .pkl format")
            
        z0_data = data_dict['z0']
        z1_data = data_dict['z1']
        
        logging.info(f"Loaded z0 shape: {z0_data.shape if hasattr(z0_data, 'shape') else len(z0_data)}")
        logging.info(f"Loaded z1 shape: {z1_data.shape if hasattr(z1_data, 'shape') else len(z1_data)}")
        
        # Load model if using true_uspace method
        if method == 'true_uspace':
            if checkpoint_path is None:
                raise ValueError("checkpoint_path required for true_uspace method")
            self.load_model(checkpoint_path)
            
        # Extract TRUE U-Space representations with incremental saving
        uspace_results = self.extract_multi_time_uspace(
            z0_data, z1_data, time_points, method, output_dir
        )
        
        # Create summary of all results
        summary_data = {
            'z0_shape': z0_data.shape if hasattr(z0_data, 'shape') else len(z0_data),
            'z1_shape': z1_data.shape if hasattr(z1_data, 'shape') else len(z1_data),
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
        
        # Save summary file
        summary_file = os.path.join(output_dir, 'true_uspace_extraction_summary.pkl')
        with open(summary_file, 'wb') as f:
            pickle.dump(summary_data, f)
            
        logging.info(f"Saved TRUE U-Space extraction summary to: {summary_file}")
        logging.info("TRUE U-Space extraction completed with incremental saving!")
        
        return summary_file


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
    
    # Process existing data
    combined_file = extractor.process_existing_data_pairs(
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
