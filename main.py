import torch
import os
from trainer import Trainer, Tester
from dataset import DataSet_hyper  # Ensure dataset is imported

def get_default_config():
    """Default configuration for the cascade model"""
    return {
        # Device settings
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'multi_gpu': torch.cuda.device_count() > 1,
        'num_workers': 4,

        # Model hyperparameters
        'timesteps': 500,         # Total diffusion steps (DDPM)
        'ddim_timesteps': 10,     # Faster steps (DDIM)
        'beta_start': 1e-4,       # Start of beta schedule
        'beta_end': 0.02,         # End of beta schedule
        'denoise_step': 50,       # Number of refinement steps

        # Training settings
        'batch_size': 64,
        'epochs': 1000,
        'lr': 2e-4,
        'weight_decay': 1e-6,
        'loss_type': 'huber',     # Loss function: 'l1', 'l2', 'smooth_l1', 'huber'

        # Checkpoint settings
        'resume': True,           # Resume from latest checkpoint
        'checkpoint_path': 'best_model.pt',  # Checkpoint filename
        'save_interval': 10,      # Save every N epochs
        'model_dir': 'models/',   # Checkpoint directory

        # Logging
        'log_dir': 'runs/diffusion_model',  # TensorBoard logs
    }


def main():
    config = get_default_config()

    # Create necessary directories
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs("./Results", exist_ok=True)

    # Select mode: 'train', 'test', 'visualize_denoise_process', 'timed_inference_test'
    mode = 'test'  # Change as needed

    if mode == 'train':
        print("Starting training...")
        trainer = Trainer(config)
        trainer.train()
    elif mode == 'test':
        print("Starting testing...")
        tester = Tester(config)
        # Choose sampling method: DDIM (faster) or DDPM (more accurate)
        tester.visualize_output(ddim=False)  # Set to True for DDIM
    elif mode == 'visualize_denoise_process':
        print("Visualizing denoising process...")
        tester = Tester(config)
        # Visualize specific samples
        for sample_idx in [5, 10, 15]:
            tester.visualize_denoise_process(
                sample_idx=sample_idx,
                ddim=False,  # Use DDPM for full process visualization
                output_dir=f"denoise_process/sample_{sample_idx}",
                steps_to_save=10  # Number of intermediate steps to save
            )
    elif mode == 'timed_inference_test':
        print("Running timed inference test...")
        tester = Tester(config)
        # Test different configurations
        test_configs = [
            {'method': 'ddpm', 'ddim_timesteps': 500},  # Full DDPM
            {'method': 'ddim', 'ddim_timesteps': 10, 'eta': 0.0},
            {'method': 'ddim', 'ddim_timesteps': 50, 'eta': 0.0},
            {'method': 'ddim', 'ddim_timesteps': 100, 'eta': 0.0},
        ]
        for cfg in test_configs:
            tester.timed_inference_test(** cfg)
    else:
        print(f"Invalid mode: {mode}. Choose 'train', 'test', 'visualize_denoise_process', or 'timed_inference_test'.")


if __name__ == "__main__":
    # Set GPU device (if multiple GPUs)
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Update with your GPU ID
    main()
