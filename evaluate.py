import os
import cv2
from tqdm import tqdm
from config import Config
from utils.metrics import EvaluationMetrics
import wandb
import pandas as pd

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
            print(f"Resumed W&B run {run_id} for evaluation.")   
    except Exception as e:
        print(f"Could not resume W&B run for evaluation. Error: {e}")

    pred_path = 'prediction_maps'
    cfg = Config()
    datasets = ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']
    
    # ✅ 1. Create a list to hold all results
    all_results = []

    for dataset in datasets:
        metric_dic = evaluate(pred_path, dataset)
        
        # Log individual metrics as charts (your original logic)
        wandb_metrics = {f"eval/{dataset}/{key}": value for key, value in metric_dic.items()}
        wandb.log(wandb_metrics)
        
        # Add dataset name to the dictionary and append to our list
        metric_dic['dataset'] = dataset
        all_results.append(metric_dic)

        print(f"\nResults for {dataset}:")
        for key, value in metric_dic.items():
            if key != 'dataset':
                print(f"{key}: {value}")

    # ✅ 2. Create and log the summary table from all results
    if all_results:
        # Create a pandas DataFrame from the list of dictionaries
        df = pd.DataFrame(all_results)
        # Reorder columns to have 'dataset' first
        df = df[['dataset'] + [col for col in df.columns if col != 'dataset']]
        # Create a W&B Table from the DataFrame
        results_table = wandb.Table(dataframe=df)
        
        print("\n--- Evaluation Summary ---")
        print(df)
        
        wandb.log({"Evaluation_Summary": results_table})

    # ✅ 3. (Optional but recommended) Log average metrics to the run's summary
    if all_results:
        avg_metrics = df.drop(columns=['dataset']).mean().to_dict()
        wandb.summary.update({f"avg_{key}": value for key, value in avg_metrics.items()})
        print("\n--- Average Metrics ---")
        for key, value in avg_metrics.items():
            print(f"avg_{key}: {value}")

    if run:
        run.finish()
        print("\nEvaluation finished and W&B run closed.")