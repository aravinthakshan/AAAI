import os
import glob
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from Model.FINet import FINet
from config import Config
from utils.dataloader_freq import TestDataset
from utils.metrics import EvaluationMetrics
from Model.LAFinet import LaplacianFINet

def inference(datasets, save_dir="prediction_maps", target_files=None):
    global model, cfg
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Extract just the filenames from the target files list
    target_filenames = set()
    if target_files:
        for file_path in target_files:
            filename = os.path.basename(file_path)
            # Remove extension and add .jpg if not present
            if not filename.endswith('.jpg'):
                filename = filename + '.jpg'
            target_filenames.add(filename)
    
    print(f"Target files to process: {target_filenames}")

    for dataset in datasets:
        assert dataset in ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']
        save_path = os.path.join(save_dir, dataset)
        os.makedirs(save_path, exist_ok=True)

        test_dataset = TestDataset(
            image_root=getattr(cfg.dp, f'test_{dataset}_imgs'),
            gt_root=getattr(cfg.dp, f'test_{dataset}_masks'),
            testsize=cfg.trainsize
        )

        processed_count = 0
        # image, gt, gt_origin, name, high, low
        for img, _, gt, name, high, low in tqdm(test_dataset, desc=f"Inference {dataset}"):
            # Skip if we have target files specified and this file is not in the list
            if target_filenames and name not in target_filenames:
                continue
                
            img = img.unsqueeze(0).cuda()
            high = high.unsqueeze(0).cuda()
            low = low.unsqueeze(0).cuda()

            out1 = model(img, high, low)[0]
            out1 = F.interpolate(out1, size=gt.shape[1:], mode='bilinear', align_corners=True)
            out1 = torch.sigmoid(out1) * 255
            out1 = out1.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)

            # save preds
            output_path = os.path.join(save_path, name)
            cv2.imwrite(output_path, out1)
            print(f"Saved: {output_path}")
            processed_count += 1

        print(f"Processed {processed_count} files for dataset {dataset}")


def evaluate(pred_path, dataset):
    global cfg
    pred_root = os.path.join(pred_path, dataset)
    metric = EvaluationMetrics()
    mask_root = getattr(cfg.dp, f'test_{dataset}_masks')
    mask_name_list = sorted(os.listdir(pred_root))

    for mask_name in tqdm(mask_name_list, desc=f"Evaluating {dataset}"):
        pred_path = os.path.join(pred_root, mask_name)
        mask_path = os.path.join(mask_root, mask_name)
        pred = cv2.imread(pred_path, flags=cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        assert pred.shape == mask.shape
        metric.step(pred=pred, gt=mask)

    return metric.get_results()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FINet Inference + Evaluation Script")
    parser.add_argument('--ckpt', type=str, default="/kaggle/input/lafinet-ffm-asf/LAFINet_ASF_epoch200.pth",
                        help="Path to a specific checkpoint. If not provided, loads the latest one.")
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['CHAMELEON', 'CAMO', 'COD10K', 'NC4K'],
                        help="Datasets to run inference on (default: all).")
    parser.add_argument('--save_dir', type=str, default="/kaggle/working/models",
                        help="Directory where checkpoints are stored.")
    parser.add_argument('--pred_dir', type=str, default="prediction_maps",
                        help="Directory to save prediction maps.")
    parser.add_argument('--model', type=str, default='FINet')
    parser.add_argument('--target_files', type=str, nargs='+', default=None,
                        help="Specific files to process (provide full paths or just filenames)")
    args = parser.parse_args()

    # Define your specific target files
    target_files = [
        "COD10K-CAM-1-Aquatic-1-BatFish-1.jpg",
        "COD10K-CAM-3-Flying-61-Katydid-4024.jpg", 
        "camourflage_00169.jpg",
        "camourflage_00160.jpg"
    ]
    
    # Use command line target files if provided, otherwise use the hardcoded ones
    if args.target_files:
        target_files = args.target_files

    # ---- Load Config & Model ----
    cfg = Config()
    
    model = FINet(backbone='efficientb0', channels=(8, 24, 32, 64)).to(cfg.device)

    if True:
        print("Initialized")
        model = LaplacianFINet(backbone='efficientb0', channels=(8, 24, 32, 64)).to(cfg.device)
    

    # ---- Load checkpoint ----
    if args.ckpt is not None:
        checkpoint = torch.load("/kaggle/input/finetmodel2/pytorch/default/1/FINet_epoch200(1).pth", map_location=cfg.device, weights_only=False)
        print(f"Loaded checkpoint from {args.ckpt}")
    if args.model == 'LAFinet':
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])

    # ---- Run Inference ----
    inference(args.datasets, save_dir=args.pred_dir, target_files=target_files)

    # ---- Run Evaluation (only on processed files) ----
    for dataset in args.datasets:
        pred_dataset_dir = os.path.join(args.pred_dir, dataset)
        if os.path.exists(pred_dataset_dir) and os.listdir(pred_dataset_dir):
            metric_dic = evaluate(args.pred_dir, dataset)
            print(f"\nResults on {dataset}:")
            print(f"SM:     {metric_dic['sm']:.4f}")
            print(f"EMean:  {metric_dic['emMean']:.4f}")
            print(f"EAdp:   {metric_dic['emAdp']:.4f}")
            print(f"WFM:    {metric_dic['wfm']:.4f}")
            print(f"MAE:    {metric_dic['mae']:.4f}")
        else:
            print(f"\nNo files processed for dataset {dataset}")


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
    parser.add_argument('--ckpt', type=str, required=True, help="Path to model checkpoint")
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