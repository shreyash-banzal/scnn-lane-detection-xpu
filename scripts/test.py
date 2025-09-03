"""
Testing script for SCNN lane detection
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import argparse
import json

from models import SCNNLaneDetection
from datasets import CULaneDataset, get_transforms
from utils import LaneDetectionMetrics, calculate_f1_score
from configs.scnn_config import vgg16_config as default_config

def test_model(model, dataloader, device):
    """Test the model and calculate metrics"""
    model.eval()
    metrics = LaneDetectionMetrics(num_classes=default_config['num_classes'])
    
    all_results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            seg_targets = batch['segmentation'].to(device)
            exist_targets = batch['existence'].to(device)
            
            predictions = model(images)
            targets = {
                'segmentation': seg_targets,
                'existence': exist_targets
            }
            
            # Update metrics
            metrics.update(predictions, targets)
            
            # Store results for detailed analysis
            seg_pred = torch.argmax(predictions['segmentation'], dim=1).cpu().numpy()
            seg_target = targets['segmentation'].cpu().numpy()
            
            for i in range(seg_pred.shape[0]):
                f1_results = calculate_f1_score(seg_pred[i], seg_target[i])
                all_results.append({
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'image_path': batch['image_path'][i],
                    'overall_f1': f1_results['overall_f1'],
                    'per_class_f1': f1_results['per_class_f1'].tolist()
                })
    
    final_metrics = metrics.compute()
    
    return final_metrics, all_results

def main():
    parser = argparse.ArgumentParser(description='Test SCNN for lane detection')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--phase', default='test', choices=['test', 'val'])
    parser.add_argument('--save_results', action='store_true', help='Save detailed results')
    args = parser.parse_args()
    
    # Device setup
    if torch.xpu.is_available():
        device = torch.device('xpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', default_config)
    
    model = SCNNLaneDetection(
        num_classes=config['num_classes'],
        backbone=config['backbone']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Dataset and DataLoader
    _, test_transform = get_transforms()
    
    test_dataset = CULaneDataset(
        config['data_root'], 
        phase=args.phase, 
        transform=test_transform,
        input_size=config['input_size']
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'],
        shuffle=False, 
        num_workers=config['num_workers']
    )
    
    print(f"Testing on {len(test_dataset)} samples")
    
    # Test model
    metrics, detailed_results = test_model(model, test_loader, device)
    
    # Print results
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Overall IoU: {metrics['overall_iou']:.4f}")
    print(f"Existence Accuracy: {metrics['existence_accuracy']:.4f}")
    print(f"Per-class IoU: {metrics['mean_iou_per_class']}")
    
    # Calculate overall F1 score from detailed results
    overall_f1_scores = [result['overall_f1'] for result in detailed_results]
    mean_f1 = sum(overall_f1_scores) / len(overall_f1_scores)
    print(f"Mean F1 Score: {mean_f1:.4f}")
    
    # Save results if requested
    if args.save_results:
        results_file = os.path.join(
            config['output_dir'], 
            f'test_results_{args.phase}.json'
        )
        
        final_results = {
            'metrics': {
                'overall_iou': float(metrics['overall_iou']),
                'existence_accuracy': float(metrics['existence_accuracy']),
                'mean_iou_per_class': metrics['mean_iou_per_class'].tolist(),
                'mean_f1_score': float(mean_f1)
            },
            'detailed_results': detailed_results,
            'config': config
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"Detailed results saved to {results_file}")

if __name__ == '__main__':
    main()
