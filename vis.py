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
from utils.dct import dct_2d
import pickle

class SingleImageProcessor:
    def __init__(self, testsize=384):
        self.testsize = testsize
        
        # Define transforms (matching TestDataset)
        self.freq_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.PILToTensor()
        ])
        
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load frequency normalization stats
        try:
            with open('./utils/freq_mean_std.pkl', 'rb') as f:
                freq_stats = pickle.load(f)
            self.freq_norm = transforms.Normalize(mean=freq_stats['mean'], std=freq_stats['std'])
        except FileNotFoundError:
            print("Warning: freq_mean_std.pkl not found. Using dummy frequency components.")
            self.freq_norm = None
    
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    
    def freq_decompose(self, freq):
        """Decompose frequency into high and low components"""
        freq_y = freq[0:64, :, :]
        freq_Cb = freq[64:128, :, :]
        freq_Cr = freq[128:192, :, :]
        
        high = torch.cat([freq_y[32:, :, :], freq_Cb[32:, :, :], freq_Cr[32:, :, :]], dim=0)
        low = torch.cat([freq_y[:32, :, :], freq_Cb[:32, :, :], freq_Cr[:32, :, :]], dim=0)
        
        return high, low
    
    def process_image(self, image_path):
        """Process single image and return tensors ready for model"""
        # Load image
        image = self.rgb_loader(image_path)
        
        # Get original size for later resizing
        original_size = image.size[::-1]  # (height, width)
        
        # Process frequency components
        if self.freq_norm is not None:
            try:
                freq = self.freq_transform(image).unsqueeze(0)
                freq = dct_2d(freq).squeeze(0)
                freq = self.freq_norm(freq) / 7.0
                high, low = self.freq_decompose(freq)
            except Exception as e:
                print(f"Warning: Frequency processing failed: {e}. Using dummy components.")
                high, low = self._create_dummy_freq_components()
        else:
            high, low = self._create_dummy_freq_components()
        
        # Process main image
        image_tensor = self.transform(image)
        
        return image_tensor, high, low, original_size
    
    def _create_dummy_freq_components(self):
        """Create dummy frequency components if DCT processing fails"""
        high = torch.zeros(96, self.testsize, self.testsize)  # 32*3 channels
        low = torch.zeros(96, self.testsize, self.testsize)   # 32*3 channels
        return high, low

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
    
    # Initialize processor
    processor = SingleImageProcessor(testsize=cfg.trainsize)
    
    # Process image
    img_tensor, high, low, original_size = processor.process_image(image_path)
    
    # Move to GPU and add batch dimension
    img_tensor = img_tensor.unsqueeze(0).cuda()
    high = high.unsqueeze(0).cuda()
    low = low.unsqueeze(0).cuda()
    
    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(img_tensor, high, low)[0]
        
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
        
    # Load config
    cfg = Config()
    
    from Model.LAFinet import LaplacianFINet
    model = LaplacianFINet(backbone='efficientb0', channels=(8, 24, 32, 64)).to(cfg.device)
    
    # Load checkpoint
    checkpoint = torch.load("/kaggle/input/finetmodel2/pytorch/default/1/FINet_epoch200(1).pth", map_location=cfg.device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    print("✓ Model loaded successfully")
    
    # Process image
    output_path = process_single_image(args.image, model, cfg, args.save_dir)
    
    print(f"\n Done! Check your output at: {output_path}")

if __name__ == '__main__':
    main()