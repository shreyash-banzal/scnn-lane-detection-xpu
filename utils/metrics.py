"""
Evaluation metrics for lane detection
"""
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

def calculate_iou(pred_mask, target_mask, num_classes=5):
    """
    Calculate Intersection over Union (IoU) for each class
    """
    ious = []
    
    for class_id in range(num_classes):
        pred_class = (pred_mask == class_id)
        target_class = (target_mask == class_id)
        
        intersection = np.logical_and(pred_class, target_class).sum()
        union = np.logical_or(pred_class, target_class).sum()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        ious.append(iou)
    
    return np.array(ious)

def calculate_f1_score(pred_mask, target_mask, num_classes=5):
    """
    Calculate F1 score for lane detection
    """
    pred_flat = pred_mask.flatten()
    target_flat = target_mask.flatten()
    
    # Calculate precision, recall, and F1 for each class
    precision, recall, f1, _ = precision_recall_fscore_support(
        target_flat, pred_flat, labels=range(num_classes), average=None, zero_division=0
    )
    
    # Calculate overall F1 (excluding background class)
    lane_classes = range(1, num_classes)  # Exclude background (class 0)
    lane_f1 = f1[lane_classes]
    overall_f1 = np.mean(lane_f1)
    
    return {
        'per_class_f1': f1,
        'lane_f1': lane_f1,
        'overall_f1': overall_f1,
        'precision': precision,
        'recall': recall
    }

def evaluate_lane_existence(pred_exist, target_exist):
    """
    Evaluate lane existence prediction
    """
    pred_exist = torch.sigmoid(pred_exist) > 0.5
    accuracy = (pred_exist == target_exist).float().mean()
    
    return {
        'existence_accuracy': accuracy.item(),
        'predictions': pred_exist.cpu().numpy(),
        'targets': target_exist.cpu().numpy()
    }

class LaneDetectionMetrics:
    """
    Comprehensive metrics calculator for lane detection
    """
    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.total_iou = np.zeros(self.num_classes)
        self.total_samples = 0
        self.existence_correct = 0
        self.existence_total = 0
    
    def update(self, predictions, targets):
        # Segmentation metrics
        seg_pred = torch.argmax(predictions['segmentation'], dim=1).cpu().numpy()
        seg_target = targets['segmentation'].cpu().numpy()
        
        batch_size = seg_pred.shape[0]
        for i in range(batch_size):
            ious = calculate_iou(seg_pred[i], seg_target[i], self.num_classes)
            self.total_iou += ious
        
        self.total_samples += batch_size
        
        # Existence metrics
        if 'existence' in predictions and 'existence' in targets:
            exist_pred = torch.sigmoid(predictions['existence']) > 0.5
            exist_target = targets['existence'].bool()
            
            correct = (exist_pred == exist_target).float().sum()
            total = exist_pred.numel()
            
            self.existence_correct += correct.item()
            self.existence_total += total
    
    def compute(self):
        mean_iou = self.total_iou / self.total_samples
        overall_iou = np.mean(mean_iou[1:])  # Exclude background
        
        existence_accuracy = self.existence_correct / max(self.existence_total, 1)
        
        return {
            'mean_iou_per_class': mean_iou,
            'overall_iou': overall_iou,
            'existence_accuracy': existence_accuracy
        }
