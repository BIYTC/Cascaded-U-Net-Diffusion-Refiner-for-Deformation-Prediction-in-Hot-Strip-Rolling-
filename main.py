import torch
import os
from trainer import Trainer, Tester

def get_default_config():
    return {
        # Device settings
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'multi_gpu': True,
        'num_workers': 4,

        # Model hyperparameters
        'timesteps': 500,
        'ddim_timesteps': 10,
        'beta_start': 1e-4,
        'beta_end': 0.02,
        'denoise_step': 50,  # Subsequent refinement denoising steps

        # Training settings
        'batch_size': 64,
        'epochs': 1000,
        'lr': 2e-4,
        'weight_decay': 1e-6,
        'loss_type': 'huber',  # 'l1', 'l2', 'smooth_l1', 'huber'

        # Checkpoint settings
        'resume': True,
        'checkpoint_path': 'best_model.pt',
        'save_interval': 1,
        'model_dir': 'models/',

        # Logging
        'log_dir': 'runs/diffusion_model',
    }


def main():
    config = get_default_config()

    # Train or test
    mode = 'visualize_denoise_process'
    # mode = 'test'

    if mode == 'train':
        trainer = Trainer(config)
        trainer.train()
    elif mode == 'test':
        tester = Tester(config)

        # Choose sampling method
        method = "ddpm"
        if method == 'ddim':
            eta = 0.5
            tester.visualize_output(ddim=True, eta=eta)
        else:
            tester.visualize_output(ddim=False)  # Visualize output results
    elif mode == "visualize_denoise_process":
        tester = Tester(config)
        method = "ddpm"
        # Compare different samples
        for idx in [5, 10, 15, 20, 25, 30]:
            print(f"Processing sample {idx}")
            # Quadratic for quadratic interval sampling
            tester.visualize_denoise_process(sample_idx=idx, ddim=False, output_dir=f"cascade_50/sample_{idx}", steps_to_save=config['denoise_step'])

    elif mode == "timed_inference_test":
        tester = Tester(config)
        # Test configuration list (example)
        test_configs = [
            {'method': 'ddim', 'ddim_timesteps': 5,     'eta': 0.95},
            {'method': 'ddim', 'ddim_timesteps': 10,    'eta': 0.95},
            {'method': 'ddim', 'ddim_timesteps': 20,    'eta': 0.95},
            {'method': 'ddim', 'ddim_timesteps': 50,    'eta': 0.95},
            {'method': 'ddim', 'ddim_timesteps': 100,   'eta': 0.95},
            {'method': 'ddim', 'ddim_timesteps': 200,   'eta': 0.95},
            {'method': 'ddim', 'ddim_timesteps': 400,   'eta': 0.95},
            {'method': 'ddim', 'ddim_timesteps': 500,   'eta': 0.95},
            {'method': 'ddim', 'ddim_timesteps': 500,   'eta': 1.0},
        ]

        for cfg in test_configs:
            print(f"\nTesting {cfg['method'].upper()} steps={cfg['ddim_timesteps']}" +
                  (f" eta={cfg['eta']}" if 'eta' in cfg else ""))
            tester.timed_inference_test(** cfg)
    else:
        print("Invalid mode. Please choose 'train' or 'test'.")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    main()
