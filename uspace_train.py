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
from datetime import datetime
import importlib.util

# Add ImageGeneration directory to path
sys.path.append('/home/felix/RectifiedFlow/ImageGeneration')

from extract_uspace_from_flow import USpaceExtractor
from semantic_boundary_trainer import SemanticBoundaryTrainer


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
    
    # Step 1: Generate semantic data pairs using current model
    logging.info("Step 1: Generating semantic data pairs...")
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
    
    # Step 2: Extract U-Space representations at t=0.2 (using neural ODE)
    logging.info("Step 2: Extracting U-Space representations...")
    control_time = getattr(config, 'semantic', {}).get('control_time', 0.2)
    uspace_extractor = USpaceExtractor(config, control_time=control_time)
    
    # Process data pairs to extract U-Space using neural ODE
    workdir_parent = os.path.dirname(iteration_dir)
    combined_file = uspace_extractor.process_existing_data_pairs(
        data_path=semantic_data_path,
        output_dir=workdir_parent,
        time_points=[control_time],
        method='neural_ode',
        checkpoint_path=base_model_path,
        iteration=iteration
    )
    
    uspace_dir = os.path.join(iteration_dir, 'uspace_extracted')
    logging.info(f"U-Space extraction completed, results in: {uspace_dir}")
    
    # Step 3: Train semantic boundary in U-Space using SVM
    logging.info("Step 3: Training semantic boundary in U-Space...")
    boundary_trainer = SemanticBoundaryTrainer(config)
    
    boundary_info = boundary_trainer.train_uspace_boundary(
        uspace_dir=uspace_dir,
        iteration=iteration,
        output_dir=iteration_dir
    )
    
    uspace_boundary_path = boundary_info.get('boundary_path') if boundary_info else None
    logging.info(f"Semantic boundary training completed")
    
    # Step 4: Train reflow model with U-Space control (u' = u + k * v_t)
    logging.info("Step 4: Training U-Space aware semantic reflow...")
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
