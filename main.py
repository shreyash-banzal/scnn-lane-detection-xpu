"""
Main entry point for SCNN lane detection project
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add the project root (one level up from scripts/) to sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

def main():
    parser = argparse.ArgumentParser(description='SCNN Lane Detection')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', default='vgg16', choices=['vgg16', 'resnet18'])
    train_parser.add_argument('--resume', type=str, default=None)
    train_parser.add_argument('--gpu', type=int, default=0)
    
    # Testing command
    test_parser = subparsers.add_parser('test', help='Test the model')
    test_parser.add_argument('--checkpoint', required=True)
    test_parser.add_argument('--phase', default='test', choices=['test', 'val'])
    test_parser.add_argument('--save_results', action='store_true')
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference')
    inference_parser.add_argument('--checkpoint', required=True)
    inference_parser.add_argument('--input', required=True)
    inference_parser.add_argument('--output', required=True)
    inference_parser.add_argument('--mode', choices=['image', 'video', 'batch'], default='image')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        from scripts.train import main as train_main
        sys.argv = ['train.py'] + [f'--{k}' if v is True else f'--{k}={v}' 
                                  for k, v in vars(args).items() 
                                  if k != 'command' and v is not None and v is not False]
        train_main()
    
    elif args.command == 'test':
        from scripts.test import main as test_main
        sys.argv = ['test.py'] + [f'--{k}' if v is True else f'--{k}={v}' 
                                 for k, v in vars(args).items() 
                                 if k != 'command' and v is not None and v is not False]
        test_main()
    
    elif args.command == 'inference':
        from scripts.inference import main as inference_main
        sys.argv = ['inference.py'] + [f'--{k}={v}' for k, v in vars(args).items() 
                                      if k != 'command' and v is not None]
        inference_main()
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
