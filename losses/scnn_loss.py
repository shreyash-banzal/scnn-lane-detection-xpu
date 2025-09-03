"""
Loss functions for SCNN lane detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SCNNLoss(nn.Module):
    """
    Combined loss function for SCNN lane detection
    """
    def __init__(self, seg_weight=1.0, exist_weight=0.1, ignore_index=255):
        super(SCNNLoss, self).__init__()
        self.seg_weight = seg_weight
        self.exist_weight = exist_weight
        
        # Segmentation loss with class weighting
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        # Existence loss
        self.exist_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions, targets):
        seg_pred = predictions['segmentation']
        exist_pred = predictions['existence']
        
        seg_target = targets['segmentation']
        exist_target = targets['existence']
        
        # Segmentation loss
        seg_loss = self.seg_loss(seg_pred, seg_target)
        
        # Existence loss
        exist_loss = self.exist_loss(exist_pred, exist_target)
        
        # Combined loss
        total_loss = self.seg_weight * seg_loss + self.exist_weight * exist_loss
        
        return {
            'total_loss': total_loss,
            'seg_loss': seg_loss,
            'exist_loss': exist_loss
        }

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
