import os
import glob
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from config import Config
from utils.dataloader_freq import TestDataset
from utils.metrics import EvaluationMetrics
from utils.dct import dct_2d
import pickle
from Model.model_factory import BACKBONE_CHOICES, MODEL_CHOICES, build_model, normalize_model_name

def inference(datasets, save_dir="prediction_maps"):
    global model, cfg
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    for dataset in datasets:
        assert dataset in ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']
        save_path = os.path.join(save_dir, dataset)
        os.makedirs(save_path, exist_ok=True)

        test_dataset = TestDataset(
            image_root=getattr(cfg.dp, f'test_{dataset}_imgs'),
            gt_root=getattr(cfg.dp, f'test_{dataset}_masks'),
            testsize=cfg.trainsize
        )

        # image, gt, gt_origin, name, high, low
        for img, _, gt, name, high, low in tqdm(test_dataset, desc=f"Inference {dataset}"):
            img = img.unsqueeze(0).to(cfg.device)
            high = high.unsqueeze(0).to(cfg.device)
            low = low.unsqueeze(0).to(cfg.device)

            out1 = model(img, high, low)[0]
            out1 = F.interpolate(out1, size=gt.shape[1:], mode='bilinear', align_corners=True)
            out1 = torch.sigmoid(out1) * 255
            out1 = out1.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)

            # save preds
            cv2.imwrite(os.path.join(save_path, name), out1)


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
    parser.add_argument('--model', type=str, default='FINet', choices=MODEL_CHOICES)
    parser.add_argument('--backbone', type=str, default='efficientb0', choices=BACKBONE_CHOICES,
                        help="Backbone used by the checkpoint.")
    args = parser.parse_args()

    # ---- Load Config & Model ----
    cfg = Config()
    
    args.model = normalize_model_name(args.model)
    model = build_model(args.model, backbone=args.backbone, channels=(8, 24, 32, 64)).to(cfg.device)
    print(f"Using {args.model} with {args.backbone} backbone.")
    

    # ---- Load checkpoint ----
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location=cfg.device, weights_only=False)
        print(f"Loaded checkpoint from {args.ckpt}")
    if args.model == 'LAFinet':
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict( checkpoint['model'])

    # ---- Run Inference ----
    inference(args.datasets, save_dir=args.pred_dir)

    # ---- Run Evaluation ----
    for dataset in args.datasets:
        metric_dic = evaluate(args.pred_dir, dataset)
        print(f"\nResults on {dataset}:")
        print(f"SM:     {metric_dic['sm']:.4f}")
        print(f"EMean:  {metric_dic['emMean']:.4f}")
        print(f"EAdp:   {metric_dic['emAdp']:.4f}")
        print(f"WFM:    {metric_dic['wfm']:.4f}")
        print(f"MAE:    {metric_dic['mae']:.4f}")
