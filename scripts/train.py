"""
Training script for SCNN lane detection
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

from models import SCNNLaneDetection
from datasets import CULaneDataset, get_transforms
from losses import SCNNLoss
from utils.metrics import LaneDetectionMetrics
from utils.visualization import visualize_lanes
from configs.scnn_config import vgg16_config as default_config

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    metrics = LaneDetectionMetrics(num_classes=default_config['num_classes'])
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} Training')
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        images = batch['image'].to(device)
        seg_targets = batch['segmentation'].to(device)
        exist_targets = batch['existence'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)
        
        # Loss calculation
        targets = {
            'segmentation': seg_targets,
            'existence': exist_targets
        }
        losses = criterion(predictions, targets)
        
        # Backward pass
        losses['total_loss'].backward()
        optimizer.step()
        
        # Update metrics
        metrics.update(predictions, targets)
        total_loss += losses['total_loss'].item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f"{losses['total_loss'].item():.4f}",
            'Seg': f"{losses['seg_loss'].item():.4f}",
            'Exist': f"{losses['exist_loss'].item():.4f}"
        })
    
    avg_loss = total_loss / len(dataloader)
    epoch_metrics = metrics.compute()
    
    return avg_loss, epoch_metrics

def validate_epoch(model, dataloader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    metrics = LaneDetectionMetrics(num_classes=default_config['num_classes'])
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} Validation')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            seg_targets = batch['segmentation'].to(device)
            exist_targets = batch['existence'].to(device)
            
            predictions = model(images)
            targets = {
                'segmentation': seg_targets,
                'existence': exist_targets
            }
            losses = criterion(predictions, targets)
            
            metrics.update(predictions, targets)
            total_loss += losses['total_loss'].item()
            
            pbar.set_postfix({'Loss': f"{losses['total_loss'].item():.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    epoch_metrics = metrics.compute()
    
    return avg_loss, epoch_metrics

def main():
    parser = argparse.ArgumentParser(description='Train SCNN for lane detection')
    parser.add_argument('--config', default='vgg16', choices=['vgg16', 'resnet18'])
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    args = parser.parse_args()
    
    # Configuration
    config = default_config
    
    # Device setup
    if torch.xpu.is_available():
        device = torch.device('xpu')
    elif torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Model initialization
    model = SCNNLaneDetection(
        num_classes=config['num_classes'],
        backbone=config['backbone']
    )
    model.to(device)
    
    # Dataset and DataLoader
    train_transform, val_transform = get_transforms()
    
    train_dataset = CULaneDataset(
        config['data_root'], 
        phase='train', 
        transform=train_transform,
        input_size=config['input_size']
    )
    
    val_dataset = CULaneDataset(
        config['data_root'], 
        phase='val', 
        transform=val_transform,
        input_size=config['input_size']
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True, 
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'],
        shuffle=False, 
        num_workers=config['num_workers']
    )
    
    # Loss and optimizer
    criterion = SCNNLoss(
        seg_weight=config['seg_weight'],
        exist_weight=config['exist_weight']
    )
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # TensorBoard writer
    writer = SummaryWriter(os.path.join(config['checkpoint_dir'], 'logs'))
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 50)
        
        # Training
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validation
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # Update learning rate
        scheduler.step()
        
        # Logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('IoU/Train', train_metrics['overall_iou'], epoch)
        writer.add_scalar('IoU/Val', val_metrics['overall_iou'], epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_metrics['overall_iou']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_metrics['overall_iou']:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'config': config
        }
        
        # Save regular checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            checkpoint_path = os.path.join(
                config['checkpoint_dir'], 
                f'scnn_epoch_{epoch+1}.pth'
            )
            torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint['best_val_loss'] = best_val_loss
            best_model_path = os.path.join(config['checkpoint_dir'], 'best_scnn_model.pth')
            torch.save(checkpoint, best_model_path)
            print("Saved best model!")
    
    writer.close()
    print("Training completed!")

if __name__ == '__main__':
    main()
