"""
Training script for Disentangled Camouflaged Object Detection.

Key differences from standard training:
1. Uses (K+1)-class prediction instead of binary
2. Includes disentangling loss to separate foreground from background patterns
3. Learns K background prototypes adaptively
"""

import torch
import random
import numpy as np
import torch.nn.functional as F
from utils.dataloader_freq import TrainDataset
from utils.LRScheduler import CosineDecay
from config import Config
from tqdm import tqdm
import os
import glob
import argparse
from utils.soap import SOAP
from utils.loss import structure_loss, create_mask_pyramid
from Model.DisentangledLAFinet import get_disentangled_model, MultiScaleDisentanglingLoss
import wandb


def train_disentangled(start_epoch=0, model_name="DisentangledLAFinet"):
    global model, loss_fn, train_datald, optimizer, cfg, scheduler
    
    print(f"Starting disentangled training: {model_name}")
    print(f"Number of background patterns: {args.num_bg_patterns}")
    print(f"Embedding dimension: {args.embed_dim}")
    
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        loss_fn.train()
        
        loss_iter = []
        binary_loss_iter = []
        disentangle_loss_iter = []
        contrast_loss_iter = []
        
        pbar = tqdm(train_datald, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for img, mask, high, low in pbar:
            optimizer.zero_grad()
            
            img = img.to(cfg.device)
            mask = mask.to(cfg.device)
            high = high.to(cfg.device)
            low = low.to(cfg.device)
            
            # Forward pass with disentangled outputs
            outputs = model(img, high, low, return_disentangled=True)
            
            # 1. Standard binary segmentation loss (multi-scale)
            binary_outputs = outputs['binary']
            output_shapes = [
                binary_outputs['out1'].shape[2:],
                binary_outputs['out2'].shape[2:],
                binary_outputs['out3'].shape[2:],
                binary_outputs['out4'].shape[2:]
            ]
            mask_pyramid = create_mask_pyramid(mask, output_shapes)
            
            binary_loss1 = structure_loss(binary_outputs['out1'], mask_pyramid[0])
            binary_loss2 = structure_loss(binary_outputs['out2'], mask_pyramid[1])
            binary_loss3 = structure_loss(binary_outputs['out3'], mask_pyramid[2])
            binary_loss4 = structure_loss(binary_outputs['out4'], mask_pyramid[3])
            
            binary_loss = 1.0 * binary_loss1 + 0.8 * binary_loss2 + 0.6 * binary_loss3 + 0.4 * binary_loss4
            
            # 2. Disentangling loss (multi-scale)
            disentangle_loss, disentangle_dict = loss_fn(outputs, mask)
            
            # Combined loss
            total_loss = args.lambda_binary * binary_loss + args.lambda_disentangle * disentangle_loss
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track losses
            loss_iter.append(total_loss.item())
            binary_loss_iter.append(binary_loss.item())
            disentangle_loss_iter.append(disentangle_loss.item())
            if 'scale1_contrast' in disentangle_dict:
                contrast_loss_iter.append(disentangle_dict['scale1_contrast'])
            
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'binary': f"{binary_loss.item():.4f}",
                'disent': f"{disentangle_loss.item():.4f}"
            })
        
        # Epoch summary
        avg_loss = np.mean(loss_iter)
        avg_binary = np.mean(binary_loss_iter)
        avg_disentangle = np.mean(disentangle_loss_iter)
        avg_contrast = np.mean(contrast_loss_iter) if contrast_loss_iter else 0
        current_lr = scheduler.get_lr()
        
        print(f'Epoch: {epoch + 1}, LR: {current_lr:.8f}')
        print(f'  Total Loss: {avg_loss:.6f}')
        print(f'  Binary Loss: {avg_binary:.6f}')
        print(f'  Disentangle Loss: {avg_disentangle:.6f}')
        print(f'  Contrast Loss: {avg_contrast:.6f}')
        
        # Log prototype statistics
        if hasattr(loss_fn.disentangle_loss, 'prototype_bank'):
            proto_bank = loss_fn.disentangle_loss.prototype_bank
            cluster_counts = proto_bank.cluster_counts.cpu().numpy()
            print(f'  Cluster distribution: {cluster_counts}')
            
            # Prototype similarity matrix (for monitoring separation)
            proto_sim = torch.matmul(proto_bank.prototypes, proto_bank.prototypes.T)
            proto_sim_off_diag = proto_sim - torch.eye(args.num_bg_patterns, device=proto_sim.device)
            max_sim = proto_sim_off_diag.max().item()
            print(f'  Max inter-prototype similarity: {max_sim:.4f}')
        
        # WandB logging
        wandb.log({
            "epoch": epoch + 1,
            "total_loss": avg_loss,
            "binary_loss": avg_binary,
            "disentangle_loss": avg_disentangle,
            "contrast_loss": avg_contrast,
            "learning_rate": current_lr,
        })
        
        scheduler.step()
        
        # Save checkpoints
        save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        if (epoch + 1) % 5 == 0 or epoch == cfg.epochs - 1:
            save_path = os.path.join(save_dir, f"DisentangledLAFinet_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'loss_fn': loss_fn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'args': vars(args)
            }, save_path)
            print(f"Checkpoint saved at {save_path}")
        
        # Final model artifact
        if epoch == cfg.epochs - 1:
            artifact = wandb.Artifact(
                name=f'DisentangledLAFinet-{args.num_bg_patterns}patterns-{wandb.run.name}',
                type='model',
                metadata={
                    'epoch': epoch + 1,
                    'loss': avg_loss,
                    'num_bg_patterns': args.num_bg_patterns
                }
            )
            artifact.add_file(local_path=save_path)
            wandb.run.log_artifact(artifact)
            print(f"✅ Saved final model artifact to W&B")


def visualize_prototypes(loss_fn, epoch, save_dir):
    """Visualize learned prototypes (optional)."""
    import matplotlib.pyplot as plt
    
    if not hasattr(loss_fn.disentangle_loss, 'prototype_bank'):
        return
    
    proto_bank = loss_fn.disentangle_loss.prototype_bank
    
    # Prototype similarity matrix
    prototypes = proto_bank.prototypes.cpu().detach()
    fg_proto = proto_bank.fg_prototype.cpu().detach()
    
    all_protos = torch.cat([fg_proto, prototypes], dim=0)
    sim_matrix = torch.matmul(all_protos, all_protos.T).numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(sim_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(f'Prototype Similarity Matrix (Epoch {epoch})')
    labels = ['FG'] + [f'BG-{i}' for i in range(proto_bank.num_prototypes)]
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    
    save_path = os.path.join(save_dir, f'prototypes_epoch{epoch}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Prototype visualization saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Disentangled LAFinet")
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='efficientb0',
                        choices=['efficientb0', 'tinynet-a'])
    parser.add_argument('--num_bg_patterns', type=int, default=8,
                        help='Number of background pattern prototypes (K)')
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='Embedding dimension for disentanglement')
    
    # Training arguments
    parser.add_argument('--optimizer', type=str, default='soap',
                        choices=['adam', 'sgd', 'soap'])
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'none'])
    parser.add_argument('--save_dir', type=str, default="/kaggle/working/models")
    parser.add_argument('--ckpt', type=str, default=None)
    
    # Loss weights
    parser.add_argument('--lambda_binary', type=float, default=1.0,
                        help='Weight for binary segmentation loss')
    parser.add_argument('--lambda_disentangle', type=float, default=0.5,
                        help='Weight for disentangling loss')
    
    args = parser.parse_args()
    
    # Seeding
    seed = 123456
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    cfg = Config()
    
    # WandB setup
    try:
        wandb_api_key = "4cdb0327752ba297aeb4f82dcc902d5f2e1d5eae"
        wandb.login(key=wandb_api_key)
        print("Logged into wandb successfully.")
    except Exception as e:
        print(f"Could not log in to wandb: {e}")
    
    wandb.init(
        project="FINET-Disentangled",
        entity="MRM_AAAI-student-26",
        config={
            "learning_rate": cfg.learning_rate,
            "architecture": "DisentangledLAFinet",
            "backbone": args.backbone,
            "optimizer": args.optimizer,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "num_bg_patterns": args.num_bg_patterns,
            "embed_dim": args.embed_dim,
            "lambda_binary": args.lambda_binary,
            "lambda_disentangle": args.lambda_disentangle,
        }
    )
    
    print(f"Started W&B run with ID: {wandb.run.id}")
    
    # Model and Loss
    model, loss_fn = get_disentangled_model(
        backbone=args.backbone,
        channels=(8, 24, 32, 64),
        embed_dim=args.embed_dim,
        num_bg_patterns=args.num_bg_patterns
    )
    model = model.to(cfg.device)
    loss_fn = loss_fn.to(cfg.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Data
    train_dataset = TrainDataset(
        image_root=cfg.dp.train_imgs,
        gt_root=cfg.dp.train_masks,
        trainsize=cfg.trainsize,
        edge_root=None
    )
    train_datald = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    # Optimizer (include loss_fn parameters for prototype learning)
    all_params = list(model.parameters()) + list(loss_fn.parameters())
    
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(all_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(all_params, lr=cfg.learning_rate, momentum=0.9, weight_decay=cfg.weight_decay)
    elif args.optimizer == 'soap':
        optimizer = SOAP(all_params, lr=3e-3, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)
    
    # Scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineDecay(optimizer, max_lr=cfg.learning_rate, min_lr=cfg.min_lr, max_epoch=cfg.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    
    # Resume from checkpoint
    start_epoch = 0
    if args.ckpt:
        print(f"Loading checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=cfg.device)
        model.load_state_dict(ckpt['model'], strict=False)
        if 'loss_fn' in ckpt:
            loss_fn.load_state_dict(ckpt['loss_fn'], strict=False)
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}")
    
    # Train
    train_disentangled(start_epoch=start_epoch)


