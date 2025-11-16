import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import os


class Unet_ModelConfig:
    def __init__(self,
                 model_size='base',
                 input_dim=21,
                 output_channels=2):
                     
        self.input_dim = input_dim
        self.output_channels = output_channels

        self.fc_layers = {
            'tiny': {
                'hidden_dims': [128, 256],
                'output_shape': (32, 32)
            },
            'base': {
                'hidden_dims': [256, 512, 1024],
                'output_shape': (64, 64)
            }
        }[model_size]

        self.encoder_channels = {
            'tiny': [4, 8, 16],
            'base': [4, 8, 16, 32]
        }[model_size]

        self.decoder_channels = {
            'tiny': [8, 4],
            'base': [16, 8, 4]
        }[model_size]

        self.activation = nn.ReLU
        self.norm_layer = nn.BatchNorm2d
        self.dropout = 0.2
        self.pooling = nn.MaxPool2d(2)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)


class UNet(nn.Module):
    """Base UNet architecture for initial head shape prediction"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        channels = config['channels']  # e.g., [64, 128, 256, 512]
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.encoder.append(self._conv_block(2, channels[0]))  # Input: (2,21,51)
        for i in range(1, len(channels)):
            self.encoder.append(self._conv_block(channels[i-1], channels[i]))
            self.encoder.append(nn.MaxPool2d(2))
        
        # Bottleneck
        self.bottleneck = self._conv_block(channels[-1], channels[-1] * 2)
        
        # Decoder
        self.decoder = nn.ModuleList()
        for i in reversed(range(len(channels))):
            self.decoder.append(nn.ConvTranspose2d(
                channels[i] * 2, channels[i], kernel_size=2, stride=2
            ))
            self.decoder.append(self._conv_block(channels[i] * 2, channels[i]))
        
        # Final output
        self.final_conv = nn.Conv2d(channels[0], 2, kernel_size=1)  # Output: (2,21,51)

    def _conv_block(self, in_channels, out_channels):
        """Basic convolutional block: Conv -> BN -> ReLU (x2)"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder with skip connections
        skips = []
        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool2d):
                x = layer(x)
            else:
                x = layer(x)
                skips.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        skip_idx = len(skips) - 1
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.ConvTranspose2d):
                x = layer(x)
            else:
                x = torch.cat([x, skips[skip_idx]], dim=1)
                x = layer(x)
                skip_idx -= 1
        
        return self.final_conv(x)


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

        # Create save directory if not exists
        os.makedirs(config['save_dir'], exist_ok=True)

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
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
        if isinstance(trainer.model, nn.DataParallel):
            trainer.model.module.load_state_dict(checkpoint['model_state'])
        else:
            trainer.model.load_state_dict(checkpoint['model_state'])

        trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
        return trainer
