#!/usr/bin/env python3
"""
U-Space Aware Semantic Reflow Iterative Training Pipeline

This pipeline implements complete U-Space aware training with mathematical control:
u' = u + k * v_t

Where:
- u: original U-Space representation extracted at t=0.2
- k: semantic classifier prediction strength  
- v_t: semantic boundary vector in U-Space

The pipeline performs iterative training where each iteration:
1. Generates semantic data pairs using current model
2. Extracts U-Space representations at t=0.2
3. Trains semantic boundary in U-Space using SVM
4. Trains reflow model with U-Space control (u' = u + k * v_t)
5. Uses improved model for next iteration
"""

import os
import sys
import json
import logging
import pickle
import argparse
import numpy as np
import torch
import wandb
import gc
from datetime import datetime
import importlib.util

# Add ImageGeneration directory to path
sys.path.append('/home/felix/RectifiedFlow/ImageGeneration')

from extract_uspace_from_flow import USpaceExtractor
from semantic_boundary_trainer import SemanticBoundaryTrainer


def get_adaptive_control_strength(probability, target_direction, base_strength=1.0):
    """
    Calculate adaptive control strength based on classification probability
    
    Args:
        probability (float): Current sample classification probability [0,1]
        target_direction (int): Target direction, 1 for enhance, -1 for reverse
        base_strength (float): Base control strength, default 1.0
    
    Returns:
        float: Calculated adaptive control strength k
    """
    # Scale probability to [-1, 1] range for bidirectional control
    k = base_strength * (2 * probability - 1) * target_direction
    
    # Limit k value range
    k = max(-2.0, min(2.0, k))
    return k


def determine_target_direction(current_label, desired_label):
    """
    Determine control direction
    
    Args:
        current_label (int): Current label (0 or 1)
        desired_label (int): Target label (0 or 1)
    
    Returns:
        int: target_direction (1 or -1)
    """
    if desired_label > current_label:
        return 1  # Enhance attribute
    else:
        return -1  # Reduce/reverse attribute


def print_k_statistics(k_values, probabilities, epoch):
    """Print k value statistics and log to wandb"""
    if len(k_values) > 0:
        import numpy as np
        mean_k = np.mean(k_values)
        std_k = np.std(k_values)
        min_k = np.min(k_values)
        max_k = np.max(k_values)
        
        logging.info(f"Epoch {epoch} - Adaptive strengths: Mean={mean_k:.3f}, Std={std_k:.3f}, Min={min_k:.3f}, Max={max_k:.3f}")
        
        # Log statistics for different confidence intervals
        low_prob_k = [k for k, p in zip(k_values, probabilities) if p < 0.3]
        mid_prob_k = [k for k, p in zip(k_values, probabilities) if 0.3 <= p <= 0.7]
        high_prob_k = [k for k, p in zip(k_values, probabilities) if p > 0.7]
        
        if low_prob_k: 
            logging.info(f"  Low confidence (p<0.3): Mean k={np.mean(low_prob_k):.3f}")
        if mid_prob_k: 
            logging.info(f"  Mid confidence (0.3≤p≤0.7): Mean k={np.mean(mid_prob_k):.3f}")
        if high_prob_k: 
            logging.info(f"  High confidence (p>0.7): Mean k={np.mean(high_prob_k):.3f}")
        
        # Log to wandb if available
        try:
            import wandb
            wandb.log({
                f'adaptive_k/mean': mean_k,
                f'adaptive_k/std': std_k,
                f'adaptive_k/min': min_k,
                f'adaptive_k/max': max_k,
                f'adaptive_k/low_conf_mean': np.mean(low_prob_k) if low_prob_k else 0,
                f'adaptive_k/mid_conf_mean': np.mean(mid_prob_k) if mid_prob_k else 0,
                f'adaptive_k/high_conf_mean': np.mean(high_prob_k) if high_prob_k else 0,
                'epoch': epoch
            })
        except:
            pass


def setup_logging(workdir):
    """Setup logging for the training session"""
    log_file = os.path.join(workdir, 'semantic_training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def setup_wandb(config, workdir, experiment_name):
    """Setup WandB logging"""
    try:
        import wandb
        
        # Simple config extraction - handle ml_collections.ConfigDict properly
        config_dict = {}
        if hasattr(config, 'training') and hasattr(config.training, 'batch_size'):
            config_dict['batch_size'] = config.training.batch_size
        if hasattr(config, 'training') and hasattr(config.training, 'n_iters'):
            config_dict['n_iters'] = config.training.n_iters
        if hasattr(config, 'optim') and hasattr(config.optim, 'lr'):
            config_dict['lr'] = config.optim.lr
        if hasattr(config, 'semantic') and hasattr(config.semantic, 'num_semantic_samples'):
            config_dict['num_semantic_samples'] = config.semantic.num_semantic_samples
        if hasattr(config, 'semantic') and hasattr(config.semantic, 'control_time'):
            config_dict['control_time'] = config.semantic.control_time
        if hasattr(config, 'model') and hasattr(config.model, 'nf'):
            config_dict['model_nf'] = config.model.nf
        if hasattr(config, 'data') and hasattr(config.data, 'dataset'):
            config_dict['dataset'] = config.data.dataset
        
        run = wandb.init(
            project="semantic-reflow-iterative",
            name=experiment_name,
            dir=workdir,
            config=config_dict
        )
        return run
    except Exception as e:
        logging.warning(f"WandB setup failed: {e}, skipping logging")
        return None

def validate_paths(config):
    """Validate required paths exist"""
    if hasattr(config, 'checkpoint_path') and config.checkpoint_path:
        if not os.path.exists(config.checkpoint_path):
            logging.error(f"Checkpoint path does not exist: {config.checkpoint_path}")
            return False
    return True

def save_iteration_metadata(workdir, iteration, data_stats, model_path, uspace_dir, uspace_boundary_path=None):
    """Save metadata for this iteration"""
    metadata = {
        'iteration': iteration,
        'timestamp': datetime.now().isoformat(),
        'data_stats': data_stats,
        'model_path': model_path,
        'uspace_dir': uspace_dir,
        'uspace_boundary_path': uspace_boundary_path
    }
    
    metadata_path = os.path.join(workdir, f'iteration_{iteration}', 'metadata.json')
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path

def run_single_iteration(config, workdir, iteration, base_model_path):
    """Run a single iteration of the semantic reflow pipeline"""
    logging.info(f"Starting iteration {iteration}")
    
    # Create iteration directory
    iteration_dir = os.path.join(workdir, f'iteration_{iteration}')
    os.makedirs(iteration_dir, exist_ok=True)
    
    # Step 1: Check for existing semantic data in current iteration, then generate if needed
    logging.info("Step 1: Looking for semantic data...")
    print("Step 1: Looking for semantic data...")

    # Check if semantic data already exists in current iteration
    current_semantic_dir = os.path.join(iteration_dir, 'semantic_analysis')
    semantic_data_found = False

    if os.path.exists(current_semantic_dir):
        # Check both root directory and pt_files subdirectory
        data_files = []
        
        # Check root semantic_analysis directory
        root_files = [f for f in os.listdir(current_semantic_dir) 
                     if f.endswith('.pt') or f.endswith('.pkl')]
        if root_files:
            data_files.extend(root_files)
            
        # Check pt_files subdirectory  
        pt_files_dir = os.path.join(current_semantic_dir, 'pt_files')
        if os.path.exists(pt_files_dir):
            pt_files = [f for f in os.listdir(pt_files_dir) 
                       if f.endswith('.pt') or f.endswith('.pkl')]
            if pt_files:
                # Use pt_files directory as the semantic data path
                semantic_data_path = pt_files_dir
                semantic_data_found = True
                logging.info(f"Found existing semantic data in pt_files: {semantic_data_path}")
                print(f"Found existing semantic data in pt_files: {semantic_data_path}")
                logging.info(f"Found {len(pt_files)} data files in pt_files: {pt_files[:5]}...")  # Show first 5 files
                
        # If we found files in root but not in pt_files, use root
        if not semantic_data_found and root_files:
            semantic_data_path = current_semantic_dir
            semantic_data_found = True
            logging.info(f"Found existing semantic data in root: {semantic_data_path}")
            print(f"Found existing semantic data in root: {semantic_data_path}")
            logging.info(f"Found {len(root_files)} data files: {root_files}")
            
        # Also check for generated_data directory with generated_pairs.pt
        if not semantic_data_found:
            generated_data_dir = os.path.join(iteration_dir, 'generated_data')
            if os.path.exists(generated_data_dir):
                generated_files = [f for f in os.listdir(generated_data_dir) 
                                 if f.endswith('.pt') or f.endswith('.pkl')]
                if generated_files:
                    semantic_data_path = generated_data_dir
                    semantic_data_found = True
                    logging.info(f"Found existing semantic data in generated_data: {semantic_data_path}")
                    logging.info(f"Found {len(generated_files)} data files: {generated_files}")

    if not semantic_data_found:
        # Generate new semantic data pairs
        logging.info("Generating new semantic data pairs...")
        print("Generating new semantic data pairs...")
        try:
            from run_lib_semantic_reflow import generate_semantic_data_pairs
            
            # Get parameters from config
            interfacegan_path = config.semantic.interfacegan_model_path
            num_samples = config.semantic.num_semantic_samples
            
            semantic_data_path = generate_semantic_data_pairs(
                config=config,
                workdir=iteration_dir,
                interfacegan_model_path=interfacegan_path,
                num_samples=num_samples
            )
            logging.info(f"Generated new semantic data path: {semantic_data_path}")
            
            # Verify the generated data
            if os.path.exists(semantic_data_path):
                generated_files = [f for f in os.listdir(semantic_data_path) 
                                if f.endswith('.pt') or f.endswith('.pkl')]
                logging.info(f"Generated {len(generated_files)} data files: {generated_files}")
            else:
                raise FileNotFoundError(f"Generated semantic data path does not exist: {semantic_data_path}")
                
        except ImportError as e:
            logging.error(f"Failed to import generate_semantic_data_pairs: {e}")
            raise
        except Exception as e:
            logging.error(f"Failed to generate semantic data: {e}")
            raise


    # Step 2: Extract U-Space representations at t=0.2 
    logging.info("Step 2: Checking for existing U-Space representations...")
    print("Step 2: Checking for existing U-Space representations...")
    uspace_dir = os.path.join(iteration_dir, 'uspace_extracted')
    control_time = getattr(config, 'semantic', {}).get('control_time', 0.2)
    uspace_file = os.path.join(uspace_dir, f'uspace_t_{control_time:.2f}.npy')
    
    uspace_exists = False
    
    # Check if U-Space data already exists
    if os.path.exists(uspace_dir):
        # Check for the specific time point file
        if os.path.exists(uspace_file):
            try:
                # Try to load and verify the file
                test_data = np.load(uspace_file, allow_pickle=True)
                if len(test_data) > 0:
                    uspace_exists = True
                    logging.info(f"Found existing U-Space data: {uspace_file}")
                    logging.info(f"U-Space data shape: {test_data.shape}")
                else:
                    logging.warning(f"U-Space file exists but is empty: {uspace_file}")
            except Exception as e:
                logging.warning(f"U-Space file exists but cannot be loaded: {e}")
        
        # If specific file doesn't exist, check for any uspace files
        if not uspace_exists:
            import glob
            existing_uspace_files = glob.glob(os.path.join(uspace_dir, 'uspace_t_*.npy'))
            if existing_uspace_files:
                logging.info(f"Found existing U-Space files: {existing_uspace_files}")
                # Use the first available file
                uspace_file = existing_uspace_files[0]
                try:
                    test_data = np.load(uspace_file, allow_pickle=True)
                    if len(test_data) > 0:
                        uspace_exists = True
                        logging.info(f"Using existing U-Space data: {uspace_file}")
                        logging.info(f"U-Space data shape: {test_data.shape}")
                except Exception as e:
                    logging.warning(f"Cannot load existing U-Space file {uspace_file}: {e}")
    
    if not uspace_exists:
        # Extract new U-Space representations
        logging.info("Extracting new U-Space representations from UNet bottleneck...")
        uspace_extractor = USpaceExtractor(config, control_time=control_time)
        
        # Process data pairs to extract U-Space using UNet bottleneck features (memory-efficient)
        workdir_parent = os.path.dirname(iteration_dir)
        combined_file = uspace_extractor.process(
            data_path=semantic_data_path,
            output_dir=workdir_parent,
            time_points=[control_time],
            checkpoint_path=base_model_path,
            iteration=iteration
        )
        
        logging.info(f"U-Space extraction completed, results in: {uspace_dir}")
        
        # Verify the extraction was successful
        if os.path.exists(uspace_file):
            try:
                test_data = np.load(uspace_file, allow_pickle=True)
                logging.info(f"Verified new U-Space data shape: {test_data.shape}")
            except Exception as e:
                logging.error(f"Failed to verify extracted U-Space data: {e}")
                raise
        else:
            logging.error(f"U-Space extraction failed - file not found: {uspace_file}")
            raise FileNotFoundError(f"U-Space extraction failed")
    else:
        logging.info("Using existing U-Space data, skipping extraction")
    
    
    # Step 3: Train semantic boundary in U-Space using SVM
    logging.info("Step 3: Training semantic boundary in U-Space...")
    print("Step 3: Training semantic boundary in U-Space...")
    boundary_trainer = SemanticBoundaryTrainer(
        interfacegan_model_path=getattr(config.semantic, 'interfacegan_model_path', ''),
        device=str(config.device)
    )
    boundary_info = boundary_trainer.train_uspace_boundary(
        uspace_dir=uspace_dir,
        iteration=iteration,
        output_dir=iteration_dir
    )
    
    uspace_boundary_path = boundary_info.get('boundary_path') if boundary_info else None
    logging.info(f"Semantic boundary training completed")
    
    # Step 4: Train reflow model with U-Space control (u' = u + k * v_t)
    logging.info("Step 4: Training U-Space aware semantic reflow...")
    print("Step 4: Training U-Space aware semantic reflow...")
    from run_lib_semantic_reflow import train_semantic_reflow
    
    model_path = train_semantic_reflow(
        config=config,
        workdir=iteration_dir,
        semantic_data_path=semantic_data_path
    )
    
    # Get data statistics
    data_stats = {
        'semantic_data_samples': len(os.listdir(semantic_data_path)) if os.path.exists(semantic_data_path) else 0,
        'uspace_samples': len(os.listdir(uspace_dir)) if os.path.exists(uspace_dir) else 0
    }
    
    # Save iteration metadata
    metadata_path = save_iteration_metadata(
        workdir, iteration, data_stats, model_path, uspace_dir, uspace_boundary_path
    )
    
    logging.info(f"Iteration {iteration} completed successfully")
    logging.info(f"Model saved at: {model_path}")
    logging.info(f"Metadata saved at: {metadata_path}")
    
    return {
        'model_path': model_path,
        'uspace_dir': uspace_dir,
        'uspace_boundary_path': uspace_boundary_path,
        'iteration_dir': iteration_dir,
        'metadata_path': metadata_path
    }

def run_iterative_training_pipeline(config_path, workdir, num_iterations=2, initial_model=None, resume_from=None):
    """Run the complete iterative training pipeline"""
    
    # Load configuration
    config_spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config_module)
    config = config_module.get_config()  # Call get_config() to get the actual config
    
    # Create workdir
    os.makedirs(workdir, exist_ok=True)
    
    # Setup logging
    setup_logging(workdir)
    logging.info("Starting U-Space Aware Semantic Reflow Iterative Training Pipeline")
    logging.info(f"Configuration: {config_path}")
    logging.info(f"Working directory: {workdir}")
    logging.info(f"Number of iterations: {num_iterations}")
    
    # Check for existing results and determine resume point
    results_path = os.path.join(workdir, 'final_results.json')
    existing_results = []
    start_iteration = 1
    current_model_path = initial_model
    
    if resume_from:
        start_iteration = resume_from
        logging.info(f"Resuming from iteration {resume_from}")
        
        # Try to load existing results
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    existing_data = json.load(f)
                    existing_results = existing_data.get('iterations', [])
                    
                # Find the last completed iteration before resume point
                last_completed = None
                for result in existing_results:
                    if result.get('iteration_dir', '').endswith(f'iteration_{resume_from - 1}'):
                        last_completed = result
                        break
                
                if last_completed and os.path.exists(last_completed['model_path']):
                    current_model_path = last_completed['model_path']
                    logging.info(f"Found previous model: {current_model_path}")
                else:
                    logging.warning(f"Could not find model from iteration {resume_from - 1}, using initial model")
            except Exception as e:
                logging.warning(f"Could not load existing results: {e}")
    else:
        # Auto-detect resume point
        iteration_dirs = [d for d in os.listdir(workdir) if d.startswith('iteration_') and os.path.isdir(os.path.join(workdir, d))]
        if iteration_dirs:
            completed_iterations = []
            for d in iteration_dirs:
                try:
                    iter_num = int(d.split('_')[1])
                    metadata_path = os.path.join(workdir, d, 'metadata.json')
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            if os.path.exists(metadata['model_path']):
                                completed_iterations.append((iter_num, metadata['model_path']))
                except:
                    continue
            
            if completed_iterations:
                completed_iterations.sort()
                last_iter, last_model = completed_iterations[-1]
                start_iteration = last_iter + 1
                current_model_path = last_model
                logging.info(f"Auto-resuming from iteration {start_iteration} with model: {current_model_path}")
    
    # Validate configuration
    if not validate_paths(config):
        logging.error("Configuration validation failed")
        return None
    
    # Use initial model or config checkpoint if no resume model found
    if not current_model_path and hasattr(config, 'reflow') and hasattr(config.reflow, 'last_flow_ckpt'):
        current_model_path = config.reflow.last_flow_ckpt
    
    if not current_model_path:
        logging.error("No initial model provided. Please specify --initial_model or set reflow.last_flow_ckpt in config")
        return None
    
    if not os.path.exists(current_model_path):
        logging.error(f"Initial model file does not exist: {current_model_path}")
        return None
    
    logging.info(f"Using model: {current_model_path}")
    
    # Setup WandB
    experiment_name = f"semantic_reflow_iterative_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if start_iteration > 1:
        experiment_name += f"_resume_{start_iteration}"
    wandb_run = setup_wandb(config, workdir, experiment_name)
    
    results = existing_results.copy()
    
    # Run iterations
    for iteration in range(start_iteration, num_iterations + 1):
        try:
            logging.info(f"Running iteration {iteration}/{num_iterations}")
            iteration_result = run_single_iteration(
                config=config,
                workdir=workdir,
                iteration=iteration,
                base_model_path=current_model_path
            )
            
            results.append(iteration_result)
            
            # Update current model for next iteration
            current_model_path = iteration_result['model_path']
            
            # Save progress after each iteration
            progress_results = {
                'num_iterations_completed': len([r for r in results if 'model_path' in r]),
                'current_model_path': current_model_path,
                'iterations': results,
                'experiment_name': experiment_name,
                'last_completed_iteration': iteration
            }
            
            with open(results_path, 'w') as f:
                json.dump(progress_results, f, indent=2)
                
            logging.info(f"Iteration {iteration}/{num_iterations} completed, progress saved")
            
        except Exception as e:
            logging.error(f"Error in iteration {iteration}: {str(e)}")
            break
    
    # Save final results
    final_results = {
        'num_iterations_completed': len([r for r in results if 'model_path' in r]),
        'final_model_path': current_model_path,
        'iterations': results,
        'experiment_name': experiment_name
    }
    
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logging.info("Pipeline completed successfully")
    logging.info(f"Final results saved to: {results_path}")
    
    if wandb_run:
        wandb_run.finish()
    
    return final_results

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run U-Space Aware Semantic Reflow Iterative Training')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--workdir', required=True, help='Working directory for training')
    parser.add_argument('--iterations', type=int, default=2, help='Number of iterations')
    parser.add_argument('--initial_model', help='Path to initial model (optional)')
    parser.add_argument('--resume_from', type=int, help='Resume from specific iteration (optional)')
    
    args = parser.parse_args()
    
    # Run the pipeline
    results = run_iterative_training_pipeline(
        config_path=args.config,
        workdir=args.workdir,
        num_iterations=args.iterations,
        initial_model=args.initial_model,
        resume_from=args.resume_from
    )
    
    if results:
        print(f"Pipeline completed successfully!")
        print(f"Final model: {results['final_model_path']}")
    else:
        print("Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
