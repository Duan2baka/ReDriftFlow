import ml_collections
import torch

def get_config():
    config = ml_collections.ConfigDict()
    
    # Training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 2  # Reduced for memory efficiency
    training.eval_batch_size = 4  # Reduced from 16
    training.n_epochs = 100
    training.n_iters = 50000  # Reduced for faster iterations
    training.snapshot_freq = 2500  # More frequent snapshots
    training.log_freq = 50
    training.eval_freq = 500  # More frequent evaluation
    training.reduce_mean = True
    training.likelihood_weighting = False
    training.continuous = False
    training.sde = 'rectified_flow'
    
    # Sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.method = 'rectified_flow'
    sampling.predictor = 'euler'
    sampling.corrector = 'none'
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.15
    sampling.sample_N = 10  # Fast sampling for iterative training
    sampling.init_type = 'gaussian'
    sampling.init_noise_scale = 1.0
    sampling.use_ode_sampler = 'rk45'
    sampling.ode_tol = 1e-3  # Relaxed for speed
    sampling.sigma_variance = 0.0
    
    # Data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'CELEBA_256'
    data.image_size = 256
    data.channels = 3
    data.num_channels = 3
    data.centered = True
    data.uniform_dequantization = False
    data.num_epochs = 100000
    
    # Model
    config.model = model = ml_collections.ConfigDict()
    model.sigma_min = 0.01
    model.sigma_max = 50
    model.num_scales = 2000  # Checkpoint actually uses 2000 scales
    model.beta_min = 0.1
    model.beta_max = 20.
    model.dropout = 0.1
    model.embedding_type = 'fourier'
    
    # UNet model - matching checkpoint architecture exactly (CelebA-HQ style)
    model.name = 'ncsnpp'
    model.scale_by_sigma = True
    model.ema_rate = 0.999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 128  # Checkpoint was trained with nf=128
    model.ch_mult = (1, 1, 2, 2, 2, 2, 2)  # CelebA-HQ uses 7 levels
    model.num_res_blocks = 2
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'output_skip'  # CelebA-HQ uses output_skip
    model.progressive_input = 'input_skip'  # CelebA-HQ uses input_skip
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'  # CelebA-HQ specifies this
    model.init_scale = 0.
    model.fourier_scale = 16
    model.conv_size = 3
    
    # Reflow configuration
    config.reflow = reflow = ml_collections.ConfigDict()
    reflow.reflow_type = 'generate_data'  # Options: 'generate_data', 'train', 'train_online'
    reflow.reflow_t_schedule = 'uniform'
    reflow.reflow_loss = 'l2'
    reflow.last_flow_ckpt = '/home/felix/Downloads/checkpoint_10.pth'
    
    # Training configuration for iterative reflow
    config.training = training = ml_collections.ConfigDict()
    training.continuous = True
    training.reduce_mean = True
    training.batch_size = 2  # Memory-safe batch size
    training.n_iters = 10000  # Reduced for faster iterations
    training.snapshot_freq = 2500  # Save more frequently
    training.log_freq = 50
    training.eval_freq = 500
    training.sampling_freq = 1000
    training.snapshot_sampling = True
    training.likelihood_weighting = False
    training.sde = 'vpsde'
    
    # Semantic configuration for iterative training
    config.semantic = semantic = ml_collections.ConfigDict()
    semantic.semantic_batch_size = 4  # Memory-safe batch size
    semantic.num_semantic_samples = 1000  # Fast testing with 1000 images
    semantic.interfacegan_model_path = "/home/felix/restyle-encoder/interfacegan_trained_model/model_smiling.pth"
    semantic.boundary_path = ""  # Will be set during training
    semantic.control_time = 0.2  # U-Space control at t=0.2
    
    # Optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 1000  # Reduced warmup for faster training
    optim.grad_clip = 1.
    
    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    # Force CUDA usage for faster training
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
    else:
        print(f"Using CUDA device: {config.device}")
    
    return config
