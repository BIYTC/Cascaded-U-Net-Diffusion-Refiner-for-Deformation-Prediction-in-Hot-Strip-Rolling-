import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import time
import os
import numpy as np
from unet.unet_main import UNetTrainer
from diffusion_model import DiffusionModel
from diffusion_utils import DiffusionUtils, ModelConfig
from dataset import DataSet_hyper


class Trainer:
    """Trainer for the cascade diffusion model"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        os.makedirs(config['model_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)

        # Initialize models
        self.model = DiffusionModel(ModelConfig()).to(self.device)
        if torch.cuda.device_count() > 1 and config['multi_gpu']:
            self.model = nn.DataParallel(self.model)

        # Initialize UNet (pretrained)
        self._init_unet()

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['epochs']
        )

        # Diffusion utils
        self.diffusion = DiffusionUtils(
            timesteps=config['timesteps'],
            ddim_timesteps=config['ddim_timesteps'],
            beta_start=config['beta_start'],
            beta_end=config['beta_end'],
            device=self.device
        )

        # Loss function
        self.loss_type = config['loss_type']

        # Data loading
        self.train_loader, self.val_loader = self._get_data_loaders()

        # Resume from checkpoint
        if config['resume'] and os.path.exists(config['model_dir'] + config['checkpoint_path']):
            self._load_checkpoint()

    def _init_unet(self):
        """Initialize pretrained UNet model"""
        model_config = {
            'channels': [64, 128, 256, 512],  # Matches UNet definition
            'num_res_blocks': 2
        }
        base_config = {
            'device': self.config['device'],
            'multi_gpu': self.config['multi_gpu'],
            'model_name': 'unet_base',
            'log_dir': './logs/unet',
            'save_dir': './checkpoints',
            'save_interval': 20,
            'batch_size': 256,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'loss_type': 'huber',
            'epochs': 100
        }
        # Load pretrained UNet
        self.unet_trainer = UNetTrainer.load_model(
            base_config, model_config, 
            os.path.join(base_config['save_dir'], 'best_unet_base.pth')
        )
        self.unet_model = self.unet_trainer.model.eval()  # Freeze UNet during diffusion training

    def _get_data_loaders(self):
        """Prepare training and validation data loaders"""
        root_path = "/home/csh/results20250331"  # Update with actual path
        all_data = np.load(f"{root_path}/train_dx_dy.npy")
        feature_stats_path = f"{root_path}/feature_stats.txt"
        global_stats_path = f"{root_path}/dx_statistics.txt"

        # Split into train/val (80/20)
        split_idx = int(0.8 * len(all_data))
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]

        train_dataset = DataSet_hyper(train_data, feature_stats_path, global_stats_path)
        val_dataset = DataSet_hyper(val_data, feature_stats_path, global_stats_path)

        return (
            DataLoader(train_dataset, batch_size=self.config['batch_size'], 
                      shuffle=True, num_workers=self.config['num_workers'], pin_memory=True),
            DataLoader(val_dataset, batch_size=self.config['batch_size'], 
                      shuffle=False, num_workers=self.config['num_workers'], pin_memory=True)
        )

    def _load_checkpoint(self):
        """Load model checkpoint"""
        checkpoint = torch.load(
            self.config['model_dir'] + self.config['checkpoint_path'],
            map_location=self.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resumed from checkpoint: {self.config['checkpoint_path']}")

    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        filename = f"epoch_{epoch}.pt" if not is_best else "best_model.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, os.path.join(self.config['model_dir'], filename))

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        with tqdm(self.train_loader, desc=f"Epoch {epoch+1}") as pbar:
            for features, targets, *_ in pbar:
                features = features.to(self.device)
                targets = targets.to(self.device)
                b = targets.shape[0]

                # Random timesteps
                t = torch.randint(0, self.config['timesteps'], (b,), device=self.device).long()

                # Compute loss
                loss = self.diffusion.p_losses(
                    self.model, targets, features, t, loss_type=self.loss_type
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    def validate(self):
        """Validate model performance"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for features, targets, *_ in self.val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                b = targets.shape[0]
                t = torch.randint(0, self.config['timesteps'], (b,), device=self.device).long()
                loss = self.diffusion.p_losses(
                    self.model, targets, features, t, loss_type=self.loss_type
                )
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self):
        """Main training loop"""
        best_val_loss = float('inf')

        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            self.scheduler.step()

            print(f"Epoch {epoch+1}/{self.config['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % self.config['save_interval'] == 0:
                self._save_checkpoint(epoch)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, is_best=True)


class Tester:
    """Tester for evaluating the cascade model"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])

        # Initialize diffusion model
        self.model = DiffusionModel(ModelConfig()).to(self.device)
        if torch.cuda.device_count() > 1 and config['multi_gpu']:
            self.model = nn.DataParallel(self.model)

        # Initialize UNet
        self._init_unet()

        # Load diffusion model weights
        self._load_model()

        # Diffusion utils
        self.diffusion = DiffusionUtils(
            timesteps=config['timesteps'],
            ddim_timesteps=config['ddim_timesteps'],
            beta_start=config['beta_start'],
            beta_end=config['beta_end'],
            device=self.device
        )

        # Load test dataset
        self.test_loader = self._get_test_loader()

    def _init_unet(self):
        """Initialize pretrained UNet (same as Trainer)"""
        model_config = {'channels': [64, 128, 256, 512], 'num_res_blocks': 2}
        base_config = {
            'device': self.config['device'],
            'multi_gpu': self.config['multi_gpu'],
            'model_name': 'unet_base',
            'log_dir': './logs/unet',
            'save_dir': './checkpoints',
            'save_interval': 20,
            'batch_size': 256,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'loss_type': 'huber',
            'epochs': 100
        }
        self.unet_trainer = UNetTrainer.load_model(
            base_config, model_config, 
            os.path.join(base_config['save_dir'], 'best_unet_base.pth')
        )
        self.unet_model = self.unet_trainer.model.eval()

    def _load_model(self):
        """Load diffusion model checkpoint"""
        checkpoint_path = os.path.join(self.config['model_dir'], self.config['checkpoint_path'])
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded diffusion model from {checkpoint_path}")

    def _get_test_loader(self):
        """Prepare test data loader"""
        root_path = "/home/csh/results20250331"  # Update with actual path
        test_data = np.load(f"{root_path}/test_dx_dy.npy")
        feature_stats_path = f"{root_path}/feature_stats.txt"
        global_stats_path = f"{root_path}/dx_statistics.txt"

        test_dataset = DataSet_hyper(test_data, feature_stats_path, global_stats_path)
        return DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

    def visualize_output(self, ddim=False, eta=0.0):
        """Visualize and save prediction results"""
        self.model.eval()
        self.unet_model.eval()

        result_dict = {
            'preds': [], 'inputs': [], 'targets': [],
            'max_dx': [], 'min_dx': [], 'max_dy': [], 'min_dy': [],
            'sample_indices': []
        }

        with torch.no_grad():
            for i, (features, target, max_dx, min_dx, max_dy, min_dy) in tqdm(enumerate(self.test_loader)):
                if i % 200 == 0:  # Save every 200th batch
                    print(f"Processing batch {i}")
                    features = features.to(self.device)
                    target = target.to(self.device)
                    b = target.shape[0]

                    # Generate cascade prediction
                    outputs = self.diffusion.sample(
                        self.model, self.unet_model,
                        features,
                        (b, 2, 21, 51),  # Output shape
                        self.device,
                        ddim=ddim,
                        eta=eta,
                        denoise_step=self.config['denoise_step']
                    )

                    # Calculate loss
                    loss = F.mse_loss(outputs, target)
                    print(f"Batch {i} Loss: {loss.item():.4f}")

                    # Store results
                    batch_indices = [i * self.test_loader.batch_size + j for j in range(b)]
                    result_dict['preds'].append(outputs.cpu().detach())
                    result_dict['inputs'].append(features.cpu().detach())
                    result_dict['targets'].append(target.cpu().detach())
                    result_dict['max_dx'].append(max_dx)
                    result_dict['min_dx'].append(min_dx)
                    result_dict['max_dy'].append(max_dy)
                    result_dict['min_dy'].append(min_dy)
                    result_dict['sample_indices'].extend(batch_indices)

                    break  # Remove to process all batches

        # Save results
        save_path = "test_results_Cascade.pth"
        torch.save({
            'preds': torch.cat(result_dict['preds'], dim=0),
            'inputs': torch.cat(result_dict['inputs'], dim=0),
            'targets': torch.cat(result_dict['targets'], dim=0),
            'max_dx': torch.cat(result_dict['max_dx'], dim=0),
            'min_dx': torch.cat(result_dict['min_dx'], dim=0),
            'max_dy': torch.cat(result_dict['max_dy'], dim=0),
            'min_dy': torch.cat(result_dict['min_dy'], dim=0),
            'sample_indices': torch.tensor(result_dict['sample_indices'])
        }, save_path)
        print(f"Results saved to {save_path}")

    def visualize_denoise_process(self, sample_idx, ddim=False, output_dir="denoise_process", steps_to_save=10):
        """Visualize the denoising process step-by-step"""
        os.makedirs(output_dir, exist_ok=True)
        self.model.eval()
        self.unet_model.eval()

        # Get single sample
        sample = self.test_loader.dataset[sample_idx]
        features, target, max_dx, min_dx, max_dy, min_dy = sample
        features = features.unsqueeze(0).to(self.device)  # Add batch dim

        # Initial UNet output
        with torch.no_grad():
            x = self.unet_model(features)
            torch.save(x.cpu(), f"{output_dir}/step_0_unet_output.pth")

            # Run denoising steps and save intermediate results
            if ddim:
                step_ratio = self.config['timesteps'] // self.config['ddim_timesteps']
                t_list = list(range(0, self.config['timesteps'], step_ratio))
                for i in reversed(range(0, self.config['ddim_timesteps'])):
                    if i % (self.config['ddim_timesteps'] // steps_to_save) == 0:
                        torch.save(x.cpu(), f"{output_dir}/step_{i}_ddim.pth")
                    t = torch.full((1,), t_list[i], device=self.device, dtype=torch.long)
                    x = self.diffusion.p_sample_ddim(self.model, x, features, t, i, eta=0.0)
            else:
                for i in reversed(range(0, self.config['timesteps'])):
                    if i % (self.config['timesteps'] // steps_to_save) == 0:
                        torch.save(x.cpu(), f"{output_dir}/step_{i}_ddpm.pth")
                    if i < self.config['denoise_step']:
                        t = torch.full((1,), i, device=self.device, dtype=torch.long)
                        x = self.diffusion.p_sample(self.model, x, features, t, i)

            # Save final output
            torch.save(x.cpu(), f"{output_dir}/final_output.pth")
            torch.save(target.cpu(), f"{output_dir}/ground_truth.pth")
        print(f"Denoising process saved to {output_dir}")

    def timed_inference_test(self, method='ddpm', ddim_timesteps=1000, eta=0.0):
        """Test inference speed and accuracy"""
        # Use a copy of diffusion utils with specified steps
        diffusion = DiffusionUtils(
            timesteps=self.config['timesteps'],
            ddim_timesteps=ddim_timesteps,
            beta_start=self.config['beta_start'],
            beta_end=self.config['beta_end'],
            device=self.device
        )

        # Random 10 samples
        total_samples = len(self.test_loader.dataset)
        indices = torch.randperm(total_samples)[:10]
        subset = Subset(self.test_loader.dataset, indices)
        test_loader = DataLoader(
            subset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
        )

        results = []
        for idx, (features, target, *_) in enumerate(test_loader):
            features = features.to(self.device)
            target = target.to(self.device)

            # Warm-up GPU
            if idx == 0:
                _ = diffusion.sample(
                    self.model, self.unet_model, features, (1, 2, 21, 51),
                    self.device, ddim=(method == 'ddim'), eta=eta
                )

            # Time inference
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
                start = time.time()
                pred = diffusion.sample(
                    self.model, self.unet_model, features, (1, 2, 21, 51),
                    self.device, ddim=(method == 'ddim'), eta=eta
                )
                torch.cuda.synchronize()
                elapsed = time.time() - start
            else:
                start = time.time()
                pred = diffusion.sample(
                    self.model, self.unet_model, features, (1, 2, 21, 51),
                    self.device, ddim=(method == 'ddim'), eta=eta
                )
                elapsed = time.time() - start

            # Metrics
            mse = F.mse_loss(pred, target).item()
            mae = F.l1_loss(pred, target).item()
            results.append((elapsed, mae, mse))
            print(f"Sample {idx+1}: Time={elapsed:.4f}s, MAE={mae:.4f}, MSE={mse:.4f}")

        # Save results
        os.makedirs("./Results", exist_ok=True)
        fname = f"./Results/{method}_steps_{ddim_timesteps}"
        if method == 'ddim':
            fname += f"_eta_{eta:.2f}"
        fname += "_results.txt"

        with open(fname, 'w') as f:
            f.write("Time (s), MAE, MSE\n")
            for res in results:
                f.write(f"{res[0]:.4f}, {res[1]:.4f}, {res[2]:.4f}\n")
            # Average
            avg_time = sum(r[0] for r in results) / len(results)
            avg_mae = sum(r[1] for r in results) / len(results)
            avg_mse = sum(r[2] for r in results) / len(results)
            f.write(f"Average, {avg_time:.4f}, {avg_mae:.4f}, {avg_mse:.4f}\n")
        print(f"Results saved to {fname}")
