import os
import re
import csv
import argparse

def parse_file(file_path):
    """Extracts metrics from a single log file."""
    metrics = {}
    with open(file_path, 'r') as f:
        content = f.read()
        
        # Regex patterns to find values
        # Matches both 'Best Validation Accuracy: 0.1234' and 'BEST_VAL_ACC: 0.1234'
        val_acc = re.search(r'(?:Best Validation Accuracy|BEST_VAL_ACC|Best Validation F1|BEST_VAL_F1):\s*([\d\.]+)', content, re.IGNORECASE)
        test_acc = re.search(r'(?:Final Test Accuracy|FINAL_TEST_ACC|Final Test F1|FINAL_TEST_F1):\s*([\d\.]+)', content, re.IGNORECASE)
        train_time = re.search(r'(?:Total Training Time|TRAIN_TIME):\s*([\d\.]+)', content, re.IGNORECASE)
        test_time = re.search(r'(?:Inference Time|TEST_TIME):\s*([\d\.]+)', content, re.IGNORECASE)

        if val_acc: metrics['Val_Acc'] = val_acc.group(1)
        if test_acc: metrics['Test_Acc'] = test_acc.group(1)
        if train_time: metrics['Train_Time'] = train_time.group(1)
        if test_time: metrics['Test_Time'] = test_time.group(1)
        
    return metrics

def collect(directory):
    results = []
    
    # Identify if we are looking at CoarseDir (based on filename structure)
    is_coarse = "coarse" in directory.lower()

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".txt"):
            continue
            
        file_path = os.path.join(directory, filename)
        metrics = parse_file(file_path)
        
        if not metrics:
            continue

        row = {"File": filename}
        
        # If it's a coarse file, extract params from filename
        # Format: Dataset_Ratio_Mask_Feat_Label_Boost_Max_Run.txt
        if is_coarse:
            parts = filename.replace(".txt", "").split("_")
            if len(parts) >= 7:
                row["Dataset"] = parts[0]
                row["Ratio"] = parts[1]
                row["Mask"] = parts[2]
                row["Feat"] = parts[3]
                row["Label"] = parts[4]
                row["Boost"] = parts[5].replace("Boost", "")
                row["Max"] = parts[6].replace("max", "")
        else:
            # Baseline format: dataset_baseline_i.txt
            row["Dataset"] = filename.split("_")[0]

        row.update(metrics)
        results.append(row)
    
    return results

def print_table(results):
    if not results:
        print("No results found.")
        return

    keys = results[0].keys()
    # Print Header
    header = " | ".join(f"{k:12}" for k in keys)
    print(header)
    print("-" * len(header))
    
    # Print Rows
    for row in results:
        print(" | ".join(f"{str(row.get(k, 'N/A')):12}" for k in keys))

def save_csv(results, output_file):
    if not results: return
    keys = results[0].keys()
    with open(output_file, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="testDir", help="Directory containing .txt logs")
    parser.add_argument("--csv", default="summary.csv", help="Output CSV filename")
    args = parser.parse_args()

    data = collect(args.dir)
    print_table(data)
    save_csv(data, args.csv)
