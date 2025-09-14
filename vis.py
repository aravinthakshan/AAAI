import os
import cv2
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from Model.FINet import FINet
from config import Config
from PIL import Image
import torchvision.transforms as transforms

def process_single_image(image_path, model, cfg, save_dir="visualization_folder"):
    """
    Process a single image through the model and save the output
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get filename for saving
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    output_filename = f"{name}_output.jpg"
    output_path = os.path.join(save_dir, output_filename)
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    # Define transforms (matching your training setup)
    transform = transforms.Compose([
        transforms.Resize((cfg.trainsize, cfg.trainsize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Transform image
    img_tensor = transform(image).unsqueeze(0).cuda()
    
    # Create dummy high and low frequency components (since we don't have them for single image)
    # You might need to adjust this based on how your dataloader creates these
    high = torch.zeros_like(img_tensor).cuda()
    low = torch.zeros_like(img_tensor).cuda()
    
    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(img_tensor, high, low)[0]
        
        # Get original image size for proper resizing
        original_size = image.size[::-1]  # PIL uses (width, height), we need (height, width)
        
        # Resize output to original image size
        output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=True)
        
        # Convert to numpy
        output = torch.sigmoid(output) * 255
        output_np = output.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)
        
        # Save output
        cv2.imwrite(output_path, output_np)
        
        print(f"✓ Processed: {image_path}")
        print(f"✓ Saved output to: {output_path}")
        
        return output_path

def main():
    parser = argparse.ArgumentParser(description="Simple Single Image Inference")
    parser.add_argument('--image', type=str, required=True, help="Path to input image")
    parser.add_argument('--model', type=str, default='FINet', choices=['FINet', 'LaFINet'], help="Model type")
    parser.add_argument('--save_dir', type=str, default="visualization_folder", help="Directory to save output")
    
    args = parser.parse_args()
    
    # Check if image file exists
    if not os.path.exists(args.image):
        print(f"Error: Image file {args.image} does not exist!")
        return
    
    # Check if checkpoint exists
    if not os.path.exists(args.ckpt):
        print(f"Error: Checkpoint file {args.ckpt} does not exist!")
        return
    
    # Load config
    cfg = Config()
    
    # Load model
    from Model.LAFinet import LaplacianFINet
    model = LaplacianFINet(backbone='efficientb0', channels=(8, 24, 32, 64)).to(cfg.device)
    
    # Load checkpoint
    checkpoint = torch.load("/kaggle/input/finetmodel2/pytorch/default/1/FINet_epoch200(1).pth", map_location=cfg.device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    print("✓ Model loaded successfully")
    
    # Process image
    output_path = process_single_image(args.image, model, cfg, args.save_dir)
    
    print(f"\n🎉 Done! Check your output at: {output_path}")

if __name__ == '__main__':
    main()