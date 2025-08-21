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

		# image, gt, gt_origin, name, high, low
		for img, _, gt, name, high, low in tqdm(test_dataset):
			img = img.unsqueeze(0).cuda()
			high = high.unsqueeze(0).cuda()
			low = low.unsqueeze(0).cuda()
			out1 = model(img, high, low)[0]
			out1 = F.interpolate(out1, size=gt.shape[1:], mode='bilinear', align_corners=True)
			out1 = torch.sigmoid(out1) * 255
			out1 = out1.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)
			# save preds
			cv2.imwrite(os.path.join(save_path, name), out1)


import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FINet Inference Script")
    parser.add_argument('--ckpt', type=str, default=None,
                        help="Path to a specific checkpoint. If not provided, loads the latest one.")
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['CHAMELEON', 'CAMO', 'COD10K', 'NC4K'],
                        help="Datasets to run inference on (default: all).")
    parser.add_argument('--save_dir', type=str, default="/kaggle/working/models",
                        help="Directory where checkpoints are stored.")
    args = parser.parse_args()

    cfg = Config()
    model = FINet(backbone='efficientb0', channels=(8, 24, 32, 64)).to(cfg.device)

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
            raise FileNotFoundError("No checkpoints found in models folder.")

    model.load_state_dict(checkpoint['model'])

    inference(args.datasets, model, cfg)

# if __name__ == '__main__':
# 	pth_path =  '/kaggle/working/models/FINet.pth'
# 	# pth_path = 'FINet-TinyNetA.pth'

# 	cfg = Config()
# 	model = FINet(backbone='efficientb0', channels=(8, 24, 32, 64)).to(cfg.device)
# 	# model = FINet(backbone='tinynet-a', channels=(8, 24, 32, 64)).to(cfg.device)
# 	model.load_state_dict(torch.load(pth_path))

# 	datasets = ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']
# 	inference(datasets)
