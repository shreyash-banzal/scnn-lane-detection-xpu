"""
Configuration management for SCNN lane detection
"""
import os
import torch

def get_config():
    """
    Get default configuration for SCNN training
    """
    config = {
        # Dataset configuration
        'data_root': 'data/CULane',
        'input_size': (288, 800),  # (height, width)
        'num_classes': 5,
        
        # Model configuration
        'backbone': 'vgg16',  # 'vgg16' or 'resnet18'
        'spatial_channels': 128,
        
        # Training configuration
        'batch_size': 8,
        'num_epochs': 50,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'momentum': 0.9,
        
        # Loss configuration
        'seg_weight': 1.0,
        'exist_weight': 0.1,
        
        # Device configuration
        'device': 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu',
        'num_workers': 4,
        
        # Checkpoint configuration
        'checkpoint_dir': 'checkpoints',
        'save_interval': 5,
        'resume': None,
        
        # Evaluation configuration
        'eval_interval': 1,
        'output_dir': 'outputs',
        
        # Visualization configuration
        'vis_samples': 5,
        'save_predictions': True,
    }
    
    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['output_dir'], exist_ok=True)
    
    return config

def update_config(config, **kwargs):
    """
    Update configuration with custom parameters
    """
    for key, value in kwargs.items():
        if key in config:
            config[key] = value
        else:
            print(f"Warning: Unknown config key '{key}' ignored")
    
    return config
