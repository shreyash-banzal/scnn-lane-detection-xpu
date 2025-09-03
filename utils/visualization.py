"""
Visualization utilities for lane detection
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

def visualize_lanes(image, lane_mask, save_path=None, show_plot=False):
    """
    Visualize lane detection results
    Args:
        image: Original image (numpy array or torch tensor)
        lane_mask: Lane segmentation mask
        save_path: Path to save the visualization
        show_plot: Whether to show the plot
    """
    # Convert to numpy if tensor
    if torch.is_tensor(image):
        if image.dim() == 4:
            image = image[0]  # Remove batch dimension
        image = image.permute(1, 2, 0).cpu().numpy()
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
    
    if torch.is_tensor(lane_mask):
        lane_mask = lane_mask.cpu().numpy()
    
    # Color map for different lanes
    colors = [
        (0, 0, 0),      # Background - Black
        (255, 0, 0),    # Lane 1 - Red
        (0, 255, 0),    # Lane 2 - Green
        (0, 0, 255),    # Lane 3 - Blue
        (255, 255, 0)   # Lane 4 - Yellow
    ]
    
    # Ensure image is in correct format
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Resize mask to match image size if needed
    if lane_mask.shape != image.shape[:2]:
        lane_mask = cv2.resize(
            lane_mask.astype(np.uint8), 
            (image.shape[1], image.shape[0]), 
            interpolation=cv2.INTER_NEAREST
        )
    
    # Create colored mask
    colored_mask = np.zeros_like(image)
    for i in range(1, min(len(colors), lane_mask.max() + 1)):
        mask = (lane_mask == i)
        colored_mask[mask] = colors[i]
    
    # Overlay on original image
    result = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
    
    if save_path:
        cv2.imwrite(save_path, result)
    
    if show_plot:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(lane_mask, cmap='tab10')
        plt.title('Lane Mask')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('Overlay Result')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return result

def save_prediction(predictions, targets, save_dir, batch_idx):
    """Save prediction results for analysis"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    seg_pred = torch.argmax(predictions['segmentation'], dim=1)
    
    for i in range(seg_pred.shape[0]):
        pred_mask = seg_pred[i].cpu().numpy()
        target_mask = targets['segmentation'][i].cpu().numpy()
        
        # Save as images
        pred_path = os.path.join(save_dir, f'batch_{batch_idx}_sample_{i}_pred.png')
        target_path = os.path.join(save_dir, f'batch_{batch_idx}_sample_{i}_target.png')
        
        cv2.imwrite(pred_path, pred_mask.astype(np.uint8) * 50)
        cv2.imwrite(target_path, target_mask.astype(np.uint8) * 50)

def create_lane_overlay_video(video_path, model, output_path):
    """Create lane detection overlay video"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame
            input_frame = cv2.resize(frame, (800, 288))
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            input_tensor = torch.from_numpy(input_frame.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            input_tensor = input_tensor.to(device)
            
            # Predict
            predictions = model(input_tensor)
            lane_mask = torch.argmax(predictions['segmentation'], dim=1)[0].cpu().numpy()
            
            # Visualize
            overlay = visualize_lanes(frame, lane_mask)
            out.write(overlay)
    
    cap.release()
    out.release()
    print(f"Video saved to {output_path}")
