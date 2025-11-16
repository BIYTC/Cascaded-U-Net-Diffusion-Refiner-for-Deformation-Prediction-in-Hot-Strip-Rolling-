import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import time
import os
import numpy as np
from unet.unet_main import UNetTrainer

class Tester:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])

        # Initialize model
        self.model = DiffusionModel(SmallConfig).to(self.device)
        # Initialize UNet model
        model_config = Unet_ModelConfig(model_size='base')
        base_config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'multi_gpu': True,
            'model_name': 'unet_base',
            'log_dir': './logs',
            'save_dir': './checkpoints',
            'save_interval': 20,
            'batch_size': 256,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'loss_type': 'huber',
            'epochs': 100
        }
        # Load model
        self.unet_model = UNetTrainer.load_model(base_config, model_config, "./checkpoints/best_unet_base.pth")

        # Load model weights
        self.load_model(config['model_dir'])

        # Initialize diffusion utils
        self.diffusion = DiffusionUtils(
            timesteps=config['timesteps'],
            ddim_timesteps = config['ddim_timesteps'],
            beta_start=config['beta_start'],
            beta_end=config['beta_end'],
            device=self.device
        )

        # Initialize dataset for ground truth comparison
        rootPath = "/home/csh/results20250331"
        test_list = np.load(f"{rootPath}/test_dx_dy.npy")
        feature_stats_path = f"{rootPath}/feature_stats.txt"
        global_stats_path = f"{rootPath}/dx_statistics.txt"
        self.test_dataset = DataSet_hyper(test_list, feature_stats_path, global_stats_path)
        # Initialize dataset and dataloader
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )

    def load_model(self, path):
        checkpoint = torch.load(path + self.config['checkpoint_path'], map_location=self.device)

        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Model loaded from {path}")

    def test_sample(self, features, ddim=False, eta=0.0):
        self.model.eval()

        with torch.no_grad():
            features = features.to(self.device)
            shape = (features.size(0), 2, 21, 51)

            # Generate samples
            samples = self.diffusion.sample(
                self.model,
                shape,
                self.device,
                ddim=ddim,
                eta=eta
            )

            return samples.cpu()

    def evaluate(self, ddim=False, eta=0.0):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            progress_bar = tqdm(self.test_loader, desc="Testing")

            for features, target, _, _, _, _ in progress_bar:
                features = features.to(self.device)
                target = target.to(self.device)

                # Generate samples
                samples = self.diffusion.sample(
                    self.model,
                    features,
                    (features.size(0), 2, 21, 51),
                    self.device,
                    ddim=ddim,
                    eta=eta
                )

                # Calculate loss
                loss = F.mse_loss(samples, target)
                total_loss += loss.item()

                progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.test_loader)
        print(f"Average test loss: {avg_loss:.4f}")
        return avg_loss

    def visualize_output(self, ddim=False, eta=0.0):
        self.model.eval()
        self.unet_model.model.eval()

        result_dict = {
            'preds': [],
            'inputs': [],
            'targets': [],
            'max_dx': [],
            'min_dx': [],
            'max_dy': [],
            'min_dy': [],
            'sample_indices': []  # Record original indices of samples in dataset
        }
        with torch.no_grad():
            progress_bar = tqdm(enumerate(self.test_loader), desc="Testing")

            for i, (features, target, max_dx, min_dx, max_dy, min_dy) in progress_bar:
                if i % 200 == 0:
                    print(f"Processing sample {i}:")
                    features = features.to(self.device)
                    target = target.to(self.device)
                    b = target.shape[0]
                    # Generate samples
                    outputs = self.diffusion.sample(
                        self.model, self.unet_model.model,
                        features,
                        (features.size(0), 2, 21, 51),
                        self.device,
                        ddim=ddim,
                        eta=eta, denoise_step=self.config['denoise_step']
                    )
                    loss = F.mse_loss(outputs, target)
                    print(f"Current sample loss: {loss}")
                    # Collect results
                    batch_indices = [i * self.test_loader.batch_size + j for j in range(b)]  # Calculate global indices
                    result_dict['preds'].append(outputs.cpu().detach())  # Store as CPU tensors
                    result_dict['inputs'].append(features.cpu().detach())
                    result_dict['targets'].append(target.cpu().detach())
                    result_dict['max_dx'].append(max_dx.cpu().detach())
                    result_dict['min_dx'].append(min_dx.cpu().detach())
                    result_dict['max_dy'].append(max_dy.cpu().detach())
                    result_dict['min_dy'].append(min_dy.cpu().detach())
                    result_dict['sample_indices'].extend(batch_indices)

                    break
        
        # Save results to file
        save_path = "test_results_Cascade_ddpm_2.pth"
        torch.save({
            'preds': torch.cat(result_dict['preds'], dim=0),  # Combine into complete tensor [N,2,21,51]
            'inputs': torch.cat(result_dict['inputs'], dim=0),  # [N,...]
            'targets': torch.cat(result_dict['targets'], dim=0),  # [N,...]
            'max_dx': torch.cat(result_dict['max_dx'], dim=0),  # [N]
            'min_dx': torch.cat(result_dict['min_dx'], dim=0),  # [N]
            'max_dy': torch.cat(result_dict['max_dy'], dim=0),  # [N]
            'min_dy': torch.cat(result_dict['min_dy'], dim=0),  # [N]
            'sample_indices': torch.tensor(result_dict['sample_indices'])
        }, save_path)
        print(f"Results saved to {save_path}")

    def timed_inference_test(self, method='ddpm', ddim_timesteps=1000, eta=0.0):
        """Perform inference test with timing statistics"""
        diffusion = DiffusionUtils(
            timesteps=self.config['timesteps'],
            ddim_timesteps = ddim_timesteps,
            beta_start=self.config['beta_start'],
            beta_end=self.config['beta_end'],
            device=self.device
        )

        # Randomly select 10 samples (non-repeating)
        total_samples = len(self.test_dataset)
        indices = torch.randperm(total_samples)[:10]
        subset = torch.utils.data.Subset(self.test_dataset, indices)
        test_loader = DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        results = []

        for idx, (features, target, *_) in enumerate(test_loader):
            # Transfer data to device
            features = features.to(self.device)
            target = target.to(self.device)

            # Warm up GPU (exclude initialization time for first run)
            if idx == 0:
                _ = diffusion.sample(self.model, features, (1, 2, 21, 51),
                                     self.device, ddim=(method == 'ddim'), eta=eta)

            # Accurate timing
            if self.device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_event.record()
            else:
                start_time = time.time()

            # Perform inference
            pred = diffusion.sample(
                self.model,
                features,
                (1, 2, 21, 51),
                self.device,
                ddim=(method.lower() == 'ddim'),
                eta=eta
            )

            # End timing
            if self.device.type == 'cuda':
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event) / 1000  # Convert ms to seconds
            else:
                end_time = time.time()
                elapsed_time = end_time - start_time

            # Calculate metrics
            mse = F.mse_loss(pred, target).item()
            mae = F.l1_loss(pred, target).item()

            results.append((elapsed_time, mae, mse))
            print(f"Sample {idx + 1}: Time={elapsed_time:.4f}s, MAE={mae:.4f}, MSE={mse:.4f}")

        # Generate filename
        fname = f"./Results/{method}_steps_{ddim_timesteps}"
        if method.lower() == 'ddim':
            fname += f"_eta_{eta:.2f}"
        fname += "_results.txt"

        # Save results
        with open(fname, 'w') as f:
