import os
import cv2
from tqdm import tqdm
from config import Config
from utils.metrics import EvaluationMetrics
import wandb
import pandas as pd
import argparse
from Model.model_factory import BACKBONE_CHOICES, MODEL_CHOICES, normalize_model_name

def wandb_is_enabled():
    run = wandb.run
    return run is not None and not getattr(run, "disabled", False)

def evaluate(pred_path, dataset):
    global cfg
    pred_root = os.path.join(pred_path, dataset)
    metric = EvaluationMetrics()
    mask_root = getattr(cfg.dp, f'test_{dataset}_masks')
    mask_name_list = sorted(os.listdir(pred_root))

    for i, mask_name in tqdm(list(enumerate(mask_name_list)), desc=f"Evaluating {dataset}"):
        pred_path = os.path.join(pred_root, mask_name)
        mask_path = os.path.join(mask_root, mask_name)
        pred = cv2.imread(pred_path, flags=cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        assert pred.shape == mask.shape
        metric.step(pred=pred, gt=mask)

    metric_dic = metric.get_results()
    return metric_dic

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate prediction maps.")
    parser.add_argument('--pred_dir', type=str, default='prediction_maps',
                        help='Directory containing dataset prediction folders.')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['CHAMELEON', 'CAMO', 'COD10K', 'NC4K'],
                        help='Datasets to evaluate.')
    parser.add_argument('--model', type=str, default='LAFinet', choices=MODEL_CHOICES,
                        help='Model name to log with evaluation metadata.')
    parser.add_argument('--backbone', type=str, default='efficientb0', choices=BACKBONE_CHOICES,
                        help='Backbone name to log with evaluation metadata.')
    parser.add_argument('--wandb_project', type=str, default=os.environ.get("WANDB_PROJECT", "FINET testing"),
                        help='W&B project name.')
    parser.add_argument('--wandb_entity', type=str, default=os.environ.get("WANDB_ENTITY", "MRM_AAAI-student-26"),
                        help='W&B entity/team name. Set to an empty string to use the default account.')
    parser.add_argument('--disable_wandb', action='store_true', default=False,
                        help='Disable W&B logging.')
    args = parser.parse_args()
    args.model = normalize_model_name(args.model)

    run = None
    if args.disable_wandb:
        run = wandb.init(mode="disabled")
    else:
        try:
            with open("wandb_run_id.txt", "r") as f:
                run_id = f.read().strip()
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                id=run_id,
                resume="must"
            )
            print(f"Resumed W&B run {run_id} for evaluation.")
        except Exception as e:
            print(f"Could not resume W&B run for evaluation; continuing with W&B disabled. Error: {e}")
            run = wandb.init(mode="disabled")

    cfg = Config()
    datasets = args.datasets
    
    all_results = []

    for dataset in datasets:
        metric_dic = evaluate(args.pred_dir, dataset)
        
        # Log individual metrics as charts (your original logic)
        wandb_metrics = {f"eval/{dataset}/{key}": value for key, value in metric_dic.items()}
        wandb_metrics.update({"eval/model": args.model, "eval/backbone": args.backbone})
        if wandb_is_enabled():
            wandb.log(wandb_metrics)
        
        # Add dataset name to the dictionary and append to our list
        metric_dic['dataset'] = dataset
        all_results.append(metric_dic)

        print(f"\nResults for {dataset}:")
        for key, value in metric_dic.items():
            if key != 'dataset':
                print(f"{key}: {value}")

    if all_results:
        # Create a pandas DataFrame from the list of dictionaries
        df = pd.DataFrame(all_results)
        # Reorder columns to have 'dataset' first
        df = df[['dataset'] + [col for col in df.columns if col != 'dataset']]
        # Create a W&B Table from the DataFrame
        results_table = wandb.Table(dataframe=df) if wandb_is_enabled() else None
        
        print("\n--- Evaluation Summary ---")
        print(df)
        
        if wandb_is_enabled():
            wandb.log({"Evaluation_Summary": results_table})

    if all_results:
        avg_metrics = df.drop(columns=['dataset']).mean().to_dict()
        if wandb_is_enabled():
            wandb.summary.update({f"avg_{key}": value for key, value in avg_metrics.items()})
        print("\n--- Average Metrics ---")
        for key, value in avg_metrics.items():
            print(f"avg_{key}: {value}")

    if wandb_is_enabled():
        run.finish()
        print("\nEvaluation finished and W&B run closed.")
