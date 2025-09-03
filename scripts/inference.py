"""
Inference script for SCNN lane detection
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import cv2
import argparse
import glob
from torchvision import transforms

from models import SCNNLaneDetection
from utils import visualize_lanes
from configs.scnn_config import vgg16_config as default_config

def load_model(checkpoint_path, device):
    """Load trained SCNN model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', default_config)
    
    model = SCNNLaneDetection(
        num_classes=config['num_classes'],
        backbone=config['backbone']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config

def preprocess_image(image_path, input_size):
    """Preprocess image for inference"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    original_image = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, input_size)
    
    # Transform to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    if len(image_resized.shape) == 3:
        input_tensor = transform(image_resized).unsqueeze(0)
    else:
        input_tensor = transform(image_resized)
    
    return input_tensor, original_image

def inference_single_image(model, image_path, output_path, config, device):
    """Run inference on a single image"""
    # Preprocess
    input_tensor, original_image = preprocess_image(image_path, config['input_size'])
    input_tensor = input_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        predictions = model(input_tensor)
        seg_pred = predictions['segmentation']
        exist_pred = predictions['existence']
        
        # Get lane mask
        lane_mask = torch.argmax(seg_pred, dim=1).cpu().numpy()[0]
        
        # Get existence probabilities
        exist_probs = torch.sigmoid(exist_pred).cpu().numpy()[0]
    
    # Visualize results
    result_image = visualize_lanes(original_image, lane_mask)
    
    # Add existence information
    text_y = 30
    for i, prob in enumerate(exist_probs):
        text = f"Lane {i+1}: {prob:.3f}"
        cv2.putText(result_image, text, (10, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        text_y += 25
    
    # Save result
    cv2.imwrite(output_path, result_image)
    print(f"Result saved to: {output_path}")
    
    return lane_mask, exist_probs

def inference_batch_images(model, input_dir, output_dir, config, device):
    """Run inference on a batch of images"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    print(f"Found {len(image_files)} images to process")
    
    for image_path in image_files:
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_result{ext}")
        
        try:
            inference_single_image(model, image_path, output_path, config, device)
            print(f"Processed: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

def inference_video(model, video_path, output_path, config, device):
    """Run inference on a video"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Preprocess frame
                input_tensor, _ = preprocess_image_array(frame, config['input_size'])
                input_tensor = input_tensor.to(device)
                
                # Predict
                predictions = model(input_tensor)
                lane_mask = torch.argmax(predictions['segmentation'], dim=1)[0].cpu().numpy()
                
                # Visualize
                result_frame = visualize_lanes(frame, lane_mask)
                out.write(result_frame)
                
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames")
                    
            except Exception as e:
                print(f"Error processing frame {frame_count}: {str(e)}")
                out.write(frame)  # Write original frame if processing fails
    
    cap.release()
    out.release()
    print(f"Video result saved to: {output_path}")

def preprocess_image_array(image_array, input_size):
    """Preprocess image array for inference"""
    original_image = image_array.copy()
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, input_size)
    
    # Transform to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image_resized).unsqueeze(0)
    
    return input_tensor, original_image

def main():
    parser = argparse.ArgumentParser(description='SCNN Lane Detection Inference')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--input', required=True, help='Input image/video/directory path')
    parser.add_argument('--output', required=True, help='Output path')
    parser.add_argument('--mode', choices=['image', 'video', 'batch'], 
                       default='image', help='Inference mode')
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
    print("Loading model...")
    model, config = load_model(args.checkpoint, device)
    print("Model loaded successfully!")
    
    # Run inference based on mode
    if args.mode == 'image':
        inference_single_image(model, args.input, args.output, config, device)
    elif args.mode == 'batch':
        inference_batch_images(model, args.input, args.output, config, device)
    elif args.mode == 'video':
        inference_video(model, args.input, args.output, config, device)

if __name__ == '__main__':
    main()
