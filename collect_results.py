import os
import re
import csv
import numpy as np
from collections import defaultdict

def parse_log_file(file_path):
    metrics = {
        'data_prep_time': None,
        'spectral_weight_time': None,
        'coarsening_exec_time': None,
        'total_preprocess_time': None,
        'best_val_acc': None,
        'final_test_acc': None,
        'train_time': None,
        'test_time': None,
        'epochs': 0
    }
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
            # Coarsening Timings
            m = re.search(r"Data Prep Time:\s+([\d.]+)", content)
            if m: metrics['data_prep_time'] = float(m.group(1))
            
            m = re.search(r"Spectral Weighting Time:\s+([\d.]+)", content)
            if m: metrics['spectral_weight_time'] = float(m.group(1))
            
            m = re.search(r"Coarsening Execution Time:\s+([\d.]+)", content)
            if m: metrics['coarsening_exec_time'] = float(m.group(1))
            
            m = re.search(r"Total Preprocessing Time:\s+([\d.]+)", content)
            if m: metrics['total_preprocess_time'] = float(m.group(1))
            
            # GNN Results
            m = re.search(r"BEST_VAL_ACC:\s+([\d.]+)", content)
            if m: metrics['best_val_acc'] = float(m.group(1))
            
            m = re.search(r"FINAL_TEST_ACC:\s+([\d.]+)", content)
            if m: metrics['final_test_acc'] = float(m.group(1))
            
            m = re.search(r"TRAIN_TIME:\s+([\d.]+)", content)
            if m: metrics['train_time'] = float(m.group(1))
            
            m = re.search(r"TEST_TIME:\s+([\d.]+)", content)
            if m: metrics['test_time'] = float(m.group(1))
            
            # Count Epochs
            metrics['epochs'] = len(re.findall(r"Epoch \d+", content))
            
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        
    return metrics

def main():
    base_dir = "LogFiles"
    output_csv = "ExperimentResultsSummary.csv"
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found.")
        return

    # Structure: results[experiment_dir][dataset][metric_name] = [list of values from runs]
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    print(f"Scanning {base_dir} for log files...")
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".log"):
                # Expecting filename format: dataset_runN.log
                match = re.match(r"(.+)_run(\d+)\.log", file)
                if match:
                    dataset = match.group(1)
                    file_path = os.path.join(root, file)
                    
                    # Experiment Dir is the parent folder
                    exp_dir = os.path.basename(root)
                    
                    metrics = parse_log_file(file_path)
                    for k, v in metrics.items():
                        if v is not None:
                            results[exp_dir][dataset][k].append(v)

    if not results:
        print("No results found to process.")
        return

    headers = [
        "Experiment", "Dataset", "Runs",
        "Prep_Time_Avg", "Prep_Time_Min", "Prep_Time_Max",
        "Spectral_Time_Avg", "Spectral_Time_Min", "Spectral_Time_Max",
        "Coarse_Time_Avg", "Coarse_Time_Min", "Coarse_Time_Max",
        "Total_Pre_Avg", "Total_Pre_Min", "Total_Pre_Max",
        "Val_Acc_Avg", "Val_Acc_Min", "Val_Acc_Max",
        "Test_Acc_Avg", "Test_Acc_Min", "Test_Acc_Max",
        "Train_Time_Avg", "Train_Time_Min", "Train_Time_Max",
        "Test_Time_Avg", "Test_Time_Min", "Test_Time_Max",
        "Epochs_Avg"
    ]

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for exp in sorted(results.keys()):
            for ds in sorted(results[exp].keys()):
                ds_metrics = results[exp][ds]
                num_runs = len(ds_metrics.get('final_test_acc', []))
                
                def get_stats(key):
                    vals = ds_metrics.get(key, [])
                    if not vals: return ["N/A"] * 3
                    return [f"{np.mean(vals):.4f}", f"{np.min(vals):.4f}", f"{np.max(vals):.4f}"]

                row = [exp, ds, num_runs]
                row.extend(get_stats('data_prep_time'))
                row.extend(get_stats('spectral_weight_time'))
                row.extend(get_stats('coarsening_exec_time'))
                row.extend(get_stats('total_preprocess_time'))
                row.extend(get_stats('best_val_acc'))
                row.extend(get_stats('final_test_acc'))
                row.extend(get_stats('train_time'))
                row.extend(get_stats('test_time'))
                
                # Epochs average only
                epochs = ds_metrics.get('epochs', [])
                row.append(f"{np.mean(epochs):.1f}" if epochs else "N/A")
                
                writer.writerow(row)

    print(f"Results successfully saved to {output_csv}")

if __name__ == "__main__":
    main()
