#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate images using a trained CelebA HQ Rectified Flow model
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

# Add the current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules
import run_lib_pytorch
import datasets
import sampling
import sde_lib
from models import utils as mutils
from models.ema import ExponentialMovingAverage

# Import the CelebA HQ config
from configs.rectified_flow.celeba_hq_pytorch_rf_gaussian import get_config


def load_model_and_config(checkpoint_path, device='cuda'):
    """Load the trained CelebA HQ model and configuration"""
    
    # Get configuration
    config = get_config()
    
    # Set device as torch.device object
    if isinstance(device, str):
        device = torch.device(device)
    config.device = device
    
    # Set sampling parameters for generation
    if not hasattr(config, 'eval'):
        from ml_collections import config_dict
        config.eval = config_dict.ConfigDict()
    config.eval.batch_size = 4  # Lower batch size for 256x256 images
    config.sampling.sample_N = 50  # Number of sampling steps (will be overridden by args)
    
    print(f"Loading CelebA HQ model from: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Image resolution: {config.data.image_size}x{config.data.image_size}")
    print(f"Default sampling steps: {config.sampling.sample_N} (can be overridden with --steps argument)")
    
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
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state['step'] = checkpoint['step']
    state['optimizer'].load_state_dict(checkpoint['optimizer'])
    state['model'].load_state_dict(checkpoint['model'], strict=False)
    state['ema'].load_state_dict(checkpoint['ema'])
    
    # Copy EMA parameters to model
    ema.copy_to(score_model.parameters())
    score_model.eval()
    
    print(f"Model loaded successfully! Training step: {state['step']}")
    
    return score_model, config, inverse_scaler


def generate_images(model, config, inverse_scaler, num_samples=4, save_dir="generated_celeba"):
    """Generate CelebA HQ images using the loaded model"""
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup SDE
    sde = sde_lib.RectifiedFlow(
        init_type=config.sampling.init_type, 
        noise_scale=config.sampling.init_noise_scale, 
        use_ode_sampler=config.sampling.use_ode_sampler, 
        sigma_var=getattr(config.sampling, 'sigma_variance', 0.0),
        ode_tol=config.sampling.ode_tol, 
        sample_N=config.sampling.sample_N
    )
    
    # Create sampling function
    sampling_shape = (num_samples, config.data.num_channels, config.data.image_size, config.data.image_size)
    sampling_eps = 1e-3
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
    
    print(f"Generating {num_samples} CelebA HQ images...")
    print(f"Image shape: {config.data.image_size}x{config.data.image_size}x{config.data.num_channels}")
    print(f"Using {config.sampling.sample_N} sampling steps")
    
    if config.sampling.sample_N == 1:
        print("⚡ One-step generation mode - very fast!")
    elif config.sampling.sample_N <= 10:
        print("🚀 Fast generation mode")
    else:
        print("🎨 High-quality generation mode")
    
    # Generate samples
    with torch.no_grad():
        print("Starting image generation...")
        samples, nfe = sampling_fn(model)
        print(f"Generation completed! Number of function evaluations: {nfe}")
    
    # Convert to numpy and rescale to [0, 255]
    samples_np = samples.permute(0, 2, 3, 1).cpu().numpy()
    samples_np = np.clip(samples_np * 255.0, 0, 255).astype(np.uint8)
    
    # Save individual images
    print("Saving images...")
    for i, img in enumerate(samples_np):
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(save_dir, f"celeba_generated_{i:04d}.png"))
        print(f"Saved image {i+1}/{len(samples_np)}")
    
    # Create a grid of images for visualization
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))
    
    if grid_size == 1:
        axes = [axes]
    elif grid_size == 2:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(grid_size * grid_size):
        if i < len(samples_np):
            axes[i].imshow(samples_np[i])
            axes[i].axis('off')
            axes[i].set_title(f'Generated Face {i+1}')
        else:
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "celeba_generated_grid.png"), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nGeneration completed!")
    print(f"Generated {len(samples_np)} CelebA HQ images saved to {save_dir}")
    print(f"Grid visualization saved as {os.path.join(save_dir, 'celeba_generated_grid.png')}")
    
    return samples_np


def main():
    parser = argparse.ArgumentParser(description='Generate CelebA HQ images using trained Rectified Flow model')
    parser.add_argument('--checkpoint', type=str, 
                       default='/home/felix/Downloads/checkpoint_10.pth',
                       help='Path to the CelebA HQ checkpoint file')
    parser.add_argument('--num_samples', type=int, default=4,
                       help='Number of images to generate (recommended: 1-8 for 256x256 images)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--save_dir', type=str, default='generated_celeba',
                       help='Directory to save generated images')
    parser.add_argument('--steps', type=int, default=50,
                       help='Number of sampling steps (1=one-step generation, 50=high quality)')
    parser.add_argument('--one_step', action='store_true',
                       help='Enable one-step generation (equivalent to --steps 1)')
    
    args = parser.parse_args()
    
    # Override steps if one_step is specified
    if args.one_step:
        args.steps = 1
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    print("="*60)
    print("CelebA HQ Image Generation with Rectified Flow")
    print("="*60)
    
    # Load model
    model, config, inverse_scaler = load_model_and_config(args.checkpoint, args.device)
    
    # Override sampling steps if specified
    config.sampling.sample_N = args.steps
    
    # Generate images
    generate_images(model, config, inverse_scaler, args.num_samples, args.save_dir)


if __name__ == "__main__":
    main()
