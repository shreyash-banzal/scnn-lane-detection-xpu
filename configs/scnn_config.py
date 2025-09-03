"""
SCNN model configuration file
"""
from utils.config import get_config, update_config

# Base configuration
config = get_config()

# Custom configuration for different experiments
vgg16_config = update_config(config.copy(), 
    backbone='vgg16',
    batch_size=8,
    learning_rate=1e-3,
    num_epochs=50
)

resnet18_config = update_config(config.copy(),
    backbone='resnet18', 
    batch_size=12,
    learning_rate=1e-3,
    num_epochs=40
)

# Configuration for different datasets
culane_config = update_config(config.copy(),
    data_root='data/CULane',
    input_size=(288, 800),
    num_classes=5
)

tusimple_config = update_config(config.copy(),
    data_root='data/TuSimple',
    input_size=(360, 640), 
    num_classes=6
)
