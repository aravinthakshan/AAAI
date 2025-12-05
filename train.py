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
from utils.loss import structure_loss, lap_structure_loss, LapLoss
import wandb
from Model.lap_utils import LaplacianPyramid

def train(start_epoch=0, model_name = "LAFinet"):
    global model, train_datald, optimizer, cfg, scheduler
    print(f"Starting training {model_name}...")
    lap_pyr_module = LaplacianPyramid(num_levels=3,c=1).to(cfg.device)
    laploss_module = LapLoss(lap_pyr_module=lap_pyr_module).to(cfg.device)
    for epoch in range(start_epoch, cfg.epochs):
        model.train()

        loss_iter = []
        for img, mask, high, low in tqdm(train_datald):
            optimizer.zero_grad()

            img = img.to(cfg.device)
            mask = mask.to(cfg.device)
            high = high.to(cfg.device)
            low = low.to(cfg.device)

            out1, out2, out3, out4 = model(img, high, low)

            if model_name == "FINet":
                loss1 = structure_loss(out1, mask)
                loss2 = structure_loss(out2, mask)
                loss3 = structure_loss(out3, mask)
                loss4 = structure_loss(out4, mask)
                loss = loss1 + loss2 + loss3 + loss4
            else:
                #loss1 = lap_structure_loss(out1, mask)
                #loss2 = lap_structure_loss(out2, mask)
                #loss3 = lap_structure_loss(out3, mask)
                #loss4 = lap_structure_loss(out4, mask)
                #loss = loss1 + loss2 + loss3 + loss4
                loss1 = structure_loss(out1, mask)
                loss2 = structure_loss(out2, mask)
                loss3 = structure_loss(out3, mask)
                loss4 = structure_loss(out4, mask)
                lap1 = laploss_module(out1.sigmoid(), mask)
                lap2 = laploss_module(out2.sigmoid(), mask)
                lap3 = laploss_module(out3.sigmoid(), mask)
                lap4 = laploss_module(out4.sigmoid(), mask)
                loss = loss1 + loss2 + loss3 + loss4 + 0.3*(lap1 + lap2 + lap3 + lap4)


            loss.backward()
            optimizer.step()
            loss_iter.append(loss.item())

        print(f'Epoch: {epoch + 1}, LR: {np.round(scheduler.get_lr(), 8)}, Loss: {np.round(np.mean(loss_iter), 8)}')

        #Wandb 
        avg_loss =  np.round(np.mean(loss_iter), 8)
        current_lr = np.round(scheduler.get_lr(), 8)
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "learning_rate": current_lr
        })
        scheduler.step()

        # Save checkpoints
        save_dir = "/kaggle/working/models"
        os.makedirs(save_dir, exist_ok=True)

        if (epoch + 1) % 5 == 0 or epoch == cfg.epochs - 1:
            save_path = os.path.join(save_dir, f"FINet_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,  # store next epoch
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, save_path)
            print(f"Checkpoint saved at {save_path}")

        #Saving weights to WandB
        if epoch == cfg.epochs - 1:
              artifact = wandb.Artifact(
                  name=f'{model_name}{cfg.epochs}-{wandb.run.name}e-final',  # Simplified name for the final model
                  type='model',
                  metadata={'epoch': epoch + 1, 'loss': np.mean(loss_iter)}
              )
              artifact.add_file(local_path=save_path)
              wandb.run.log_artifact(artifact)
              print(f"✅ Saved final model artifact to W&B")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train FINet model with options")
    parser.add_argument('--backbone', type=str, default='efficientb0',
                        choices=['efficientb0', 'tinynet-a'],
                        help='Backbone network for FINet')
    parser.add_argument('--optimizer', type=str, default='soap',
                        choices=['adam', 'sgd','soap'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--save_dir', type=str, default="/kaggle/working/models",
                        help='Directory to save checkpoints')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to a specific checkpoint to resume from')
    parser.add_argument('--model', type=str, default='FInet',)
    parser.add_argument('--trial', action='store_true', default= False)
    args = parser.parse_args()

    # ---- Seeding ----
    seed = 123456
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    cfg = Config()

    #WANDB

    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        wandb_api_key = "4cdb0327752ba297aeb4f82dcc902d5f2e1d5eae"
        wandb.login(key=wandb_api_key)
        print("Logged into wandb successfully.")
    except ImportError:
        print("Kaggle secrets not found. Please ensure you're in a Kaggle environment or log in manually.")
    except Exception as e:
        print(f"Could not log in to wandb: {e}")

    wandb.init(
        project="FINET testing",
        entity="MRM_AAAI-student-26", 
        config={
            "learning_rate": cfg.learning_rate,
            "architecture": args.model,
            "backbone": args.backbone,
            "optimizer": args.optimizer,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
        }
    )
    #saving wandb name for use inn inference and evaluate
    print(f"Started W&B run with ID: {wandb.run.id}")
    with open("wandb_run_id.txt", "w") as f:
        f.write(wandb.run.id)

    # ---- Model ----
    from Model.FINet import FINet
    from Model.LAFinet import LaplacianFINet
    if args.model == "FINet":
        model = FINet(backbone=args.backbone, channels=(8, 24, 32, 64)).to(cfg.device)
    else:
        model = LaplacianFINet(backbone=args.backbone, channels=(8, 24, 32, 64)).to(cfg.device)

    print(f"Total parameters for model '{args.model}': {sum(p.numel() for p in model.parameters()):,}")

    # ---- Data ----
    train_dataset = TrainDataset(image_root=cfg.dp.train_imgs,
                                 gt_root=cfg.dp.train_masks,
                                 trainsize=cfg.trainsize,
                                 edge_root=None)
    train_datald = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=True,
                                               num_workers=cfg.num_workers,
                                               pin_memory=True)

    # ---- Optimizer ----
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=cfg.learning_rate,
                                     weight_decay=cfg.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=cfg.learning_rate,
                                    momentum=0.9,
                                    weight_decay=cfg.weight_decay)
    elif args.optimizer == 'soap':
        optimizer = SOAP(model.parameters(), lr = 3e-3, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)

    # ---- Scheduler ----
    if args.scheduler == 'cosine':
        scheduler = CosineDecay(optimizer, max_lr=cfg.learning_rate,
                                min_lr=cfg.min_lr, max_epoch=cfg.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    # ---- Resume from checkpoint ----
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoints = sorted(glob.glob(os.path.join(args.save_dir, "LAFINet*.pth")))

    start_epoch = 0
    if checkpoints and args.trial == True:
        # latest_ckpt = checkpoints[-1]
        latest_ckpt = args.ckpt if args.ckpt else checkpoints[-1]  # last checkpoint or specified
        print(f"Resuming training from checkpoint {latest_ckpt}...")
        ckpt = torch.load(latest_ckpt, map_location=cfg.device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
    elif args.ckpt:
        latest_ckpt = args.ckpt
        print(f"Resuming training from checkpoint {latest_ckpt}...")
        ckpt = torch.load(latest_ckpt, map_location=cfg.device)
        model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
    else:
        print("No checkpoint found. Training from scratch.")

    train(start_epoch=start_epoch, model_name=args.model)
    
    
