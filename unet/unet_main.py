import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import os

class UNetTrainer:
    def __init__(self, config, model_config):
        self.config = config
        self.model_config = model_config

        # Device configuration
        self.device = torch.device(config['device'])
        self.model = self._init_model().to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.75,
            patience=5
        )

        # Loss function
        self.loss_fn = {
            'l1': nn.L1Loss(),
            'l2': nn.MSELoss(),
            'smooth_l1': nn.SmoothL1Loss(),
            'huber': nn.HuberLoss()
        }[config['loss_type']]

        # Logging
        self.writer = SummaryWriter(log_dir=config['log_dir'])

    def _init_model(self):
        model = UNet(self.model_config)
        if torch.cuda.device_count() > 1 and self.config['multi_gpu']:
            model = nn.DataParallel(model)
        return model

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        start_time = time.time()

        with tqdm(train_loader, desc="Training") as pbar:
            for features, targets, *_ in pbar:
                features = features.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        return avg_loss, epoch_time

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for features, targets, *_ in val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(features)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def test(self, test_loader):
        self.model.eval()
        # Initialize result storage structure
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
            for i, (features, target, max_dx, min_dx, max_dy, min_dy) in tqdm(enumerate(test_loader)):
                if i % 10 == 0:
                    print(f"\nProcessing batch {i}\n")
                    st = time.time()
                    features = features.to(self.device).float()
                    target = target.to(self.device).float()
                    b = target.shape[0]
                    outputs = self.model(features)
                    # Collect results
                    batch_indices = [i * test_loader.batch_size + j for j in range(b)]  # Calculate global indices
                    result_dict['preds'].append(outputs.cpu().detach())  # Store as CPU tensors
                    result_dict['inputs'].append(features.cpu().detach())
                    result_dict['targets'].append(target.cpu().detach())
                    result_dict['max_dx'].append(max_dx.cpu().detach())
                    result_dict['min_dx'].append(min_dx.cpu().detach())
                    result_dict['max_dy'].append(max_dy.cpu().detach())
                    result_dict['min_dy'].append(min_dy.cpu().detach())
                    result_dict['sample_indices'].extend(batch_indices)
                    print(f"Time taken: {time.time() - st}")
        
        # Save results to file
        save_path = "test_results.pth"
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

    def train(self, train_loader, val_loader):
        best_loss = float('inf')

        for epoch in range(self.config['epochs']):
            train_loss, train_time = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.scheduler.step(val_loss)

            # Record logs
            self.writer.add_scalars('Loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_model(f"best_{self.config['model_name']}.pth")

            # Save periodically
            if epoch % self.config['save_interval'] == 0:
                self.save_model(f"{self.config['model_name']}_epoch{epoch}.pth")

            print(f"Epoch {epoch + 1}/{self.config['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Time: {train_time:.1f}s")

    def save_model(self, filename):
        save_path = os.path.join(self.config['save_dir'], filename)
        torch.save({
            'model_state': self.model.module.state_dict() if isinstance(self.model,
                                                                       nn.DataParallel) else self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config
        }, save_path)

    @classmethod
    def load_model(cls, config, model_config, checkpoint_path):
        trainer = cls(config, model_config)
        checkpoint = torch.load(checkpoint_path)
        if isinstance(trainer.model, nn.DataParallel):
            trainer.model.module.load_state_dict(checkpoint['model_state'])
        else:
            trainer.model.load_state_dict(checkpoint['model_state'])

        trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
        return trainer
