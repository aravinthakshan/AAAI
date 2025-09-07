import os
import torch 
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import cv2
from Model.FINet import FINet
from config import Config
from utils.dataloader_freq import TestDataset
import argparse
import wandb
import glob

NUM_IMAGES_TO_LOG = 5

def inference(datasets):
    global model, cfg
    model.eval()

    for dataset in datasets:
        assert dataset in ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']
        save_path = os.path.join('prediction_maps', dataset)
        os.makedirs(save_path, exist_ok=True)

        test_dataset = TestDataset(image_root=getattr(cfg.dp, f'test_{dataset}_imgs'),
                                   gt_root=getattr(cfg.dp, f'test_{dataset}_masks'),
                                   testsize=cfg.trainsize)

        inference_table = wandb.Table(columns=["Image Name", "Image", "Ground Truth", "Prediction"])
        log_count = 0

        for img_tensor, _, gt_tensor, name, high, low in tqdm(test_dataset, desc=f"Inferring on {dataset}"):
            img_cuda = img_tensor.unsqueeze(0).cuda()
            high = high.unsqueeze(0).cuda()
            low = low.unsqueeze(0).cuda()
            
            out1_tensor = model(img_cuda, high, low)[0]
            out1_tensor = F.interpolate(out1_tensor, size=gt_tensor.shape[1:], mode='bilinear', align_corners=True)
            out1_tensor = torch.sigmoid(out1_tensor) * 255
            
            pred_np = out1_tensor.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)
            
            cv2.imwrite(os.path.join(save_path, name), pred_np)

            if log_count < NUM_IMAGES_TO_LOG:
                img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = (img_np * std + mean) * 255
                img_np = img_np.astype(np.uint8)

                gt_np = gt_tensor.squeeze().cpu().numpy().astype(np.uint8) * 255

                inference_table.add_data(
                    name,
                    wandb.Image(img_np),
                    wandb.Image(gt_np),
                    wandb.Image(pred_np)
                )
                log_count += 1
        
        wandb.log({f"Inference Results/{dataset}": inference_table})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FINet Inference Script")
    parser.add_argument('--ckpt', type=str, default=None,
                        help="Path to a specific checkpoint. If not provided, loads the latest one.")
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['CHAMELEON', 'CAMO', 'COD10K', 'NC4K'],
                        help="Datasets to run inference on (default: all).")
    parser.add_argument('--save_dir', type=str, default="/kaggle/working/models",
                        help="Directory where checkpoints are stored.")
    parser.add_argument('--model', type=str, default='FINet',)
    args = parser.parse_args()

    run = None
    try:
        with open("wandb_run_id.txt", "r") as f:
            run_id = f.read().strip()
        run = wandb.init(
            project="FINET testing",
            entity="MRM_AAAI-student-26",
            id=run_id,
            resume="must"
        )
        print(f"Resumed W&B run {run_id} for inference.")
    except Exception as e:
        print(f"Could not resume W&B run for inference. Error: {e}")

    cfg = Config()

    model = FINet(backbone='efficientb0', channels=(8, 24, 32, 64)).to(cfg.device)
    if args.model == "LaFINet":
        from Model.LAFinet import LaplacianFINet
        print("Using LaFINet model for inference.")
        model = LaplacianFINet(backbone='efficientb0', channels=(8, 24, 32, 64)).to(cfg.device)

    # ---- Load checkpoint ----
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location=cfg.device)
        print(f"Loaded checkpoint from {args.ckpt}")
    else:
        checkpoints = sorted(glob.glob(os.path.join(args.save_dir, "FINet_epoch*.pth")))

        if checkpoints:
            latest_ckpt = checkpoints[-1]
            checkpoint = torch.load(latest_ckpt, map_location=cfg.device)
            print(f"Loaded latest checkpoint: {latest_ckpt}")
        else:
            raise FileNotFoundError(f"No {args.model} checkpoints found in {args.save_dir}.")

    model.load_state_dict(checkpoint['model'])

    inference(args.datasets)

# if __name__ == '__main__':
# 	pth_path =  '/kaggle/working/models/FINet.pth'
# 	# pth_path = 'FINet-TinyNetA.pth'

# 	cfg = Config()
# 	model = FINet(backbone='efficientb0', channels=(8, 24, 32, 64)).to(cfg.device)
# 	# model = FINet(backbone='tinynet-a', channels=(8, 24, 32, 64)).to(cfg.device)
# 	model.load_state_dict(torch.load(pth_path))

# 	datasets = ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']
# 	inference(datasets)
