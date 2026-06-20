import os
import pandas as pd
import numpy as np

def aggregate_results(dataset_name, quantitative_folder, label, epochs_no, batch_size):
    if not os.path.exists(quantitative_folder):
        print(f"Error: Folder '{quantitative_folder}' does not exist.")
        return

    if dataset_name == 'loco':
        CLASSES = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
        sep = ','
    elif dataset_name == 'visa':
        CLASSES = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]
        sep = '&'
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    dfs = []
    for class_name in CLASSES:
        if dataset_name == 'loco':
            filename = f"{label}_{class_name}_{epochs_no}ep_{batch_size}bs.csv"
        else:
            filename = f"{label}_{class_name}_{epochs_no}ep_{batch_size}bs.csv"
        
        filepath = os.path.join(quantitative_folder, filename)

        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, sep=sep)
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        else:
            print(f"Warning: Result file for '{class_name}' not found at {filepath}")

    if not dfs:
        print("\nNo result files found matching the criteria. Exiting.")
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
    mean_values = combined_df[numeric_cols].mean()
    
    mean_df = pd.DataFrame(mean_values).T
    mean_df['class_name'] = 'AVERAGE'
    
    cols = ['class_name'] + [c for c in mean_df.columns if c != 'class_name']
    mean_df = mean_df[cols]
    combined_df = combined_df[cols]

    final_df = pd.concat([combined_df, mean_df], ignore_index=True)
    
    if dataset_name == 'loco':
        print_markdown_summary_loco(mean_values, label)
    else:
        print_markdown_summary_visa(mean_values)

    output_filename = f"{label}_AVERAGE_{epochs_no}ep_{batch_size}bs.csv"
    output_path = os.path.join(quantitative_folder, output_filename)
    final_df.to_csv(output_path, index=False, sep=',')

def print_markdown_summary_loco(mean_vals, label):
    get_val = lambda k: mean_vals.get(k, 0.0)
    glob_auc, glob_30, glob_10, glob_05, glob_01 = get_val('global_i_auroc'), get_val('global_spro_30'), get_val('global_spro_10'), get_val('global_spro_05'), get_val('global_spro_01')
    struct_auc, struct_30, struct_10, struct_05, struct_01 = get_val('struct_i_auroc'), get_val('struct_spro_30'), get_val('struct_spro_10'), get_val('struct_spro_05'), get_val('struct_spro_01')
    logic_auc, logic_30, logic_10, logic_05, logic_01 = get_val('logic_i_auroc'), get_val('logic_spro_30'), get_val('logic_spro_10'), get_val('logic_spro_05'), get_val('logic_spro_01')

    print(f"\n=== MEAN METRICS ({label}) ===")
    header = f"{'Type':<12} | {'I-AUROC':<8} | {'sPRO@30%':<8} | {'sPRO@10%':<8} | {'sPRO@5%':<8} | {'sPRO@1%':<8}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    print(f"{'Global':<12} | {glob_auc:.3f}    | {glob_30:.3f}    | {glob_10:.3f}    | {glob_05:.3f}    | {glob_01:.3f}")
    print(f"{'Structural':<12} | {struct_auc:.3f}    | {struct_30:.3f}    | {struct_10:.3f}    | {struct_05:.3f}    | {struct_01:.3f}")
    print(f"{'Logical':<12} | {logic_auc:.3f}    | {logic_30:.3f}    | {logic_10:.3f}    | {logic_05:.3f}    | {logic_01:.3f}")
    print("-" * len(header))

def print_markdown_summary_visa(mean_vals):
    get_val = lambda k: mean_vals.get(k, 0.0)
    p_auc = get_val('p_auroc')
    i_auc = get_val('i_auroc')

    def get_quartile_row(q):
        return [get_val(f'aupro_30_{q}'), get_val(f'aupro_10_{q}'), get_val(f'aupro_05_{q}'), get_val(f'aupro_01_{q}')]

    q4, q3, q2, q1 = get_quartile_row('q4'), get_quartile_row('q3'), get_quartile_row('q2'), get_quartile_row('q1')

    print("\n" + "="*80)
    print(f"=== MEAN METRICS (VisA) ===")
    print("="*80)
    print(f"P-AUROC  |  I-AUROC")
    print(f"  {p_auc:.3f}   |   {i_auc:.3f}\n")
    header = f" {'QUARTILE':^8} | {'AUPRO@30%':^11} | {'AUPRO@10%':^11} | {'AUPRO@5%':^10} | {'AUPRO@1%':^10} |"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    print(f" {'Q4':^8} | {q4[0]:^11.3f} | {q4[1]:^11.3f} | {q4[2]:^10.3f} | {q4[3]:^10.3f} |")
    print(f" {'Q3':^8} | {q3[0]:^11.3f} | {q3[1]:^11.3f} | {q3[2]:^10.3f} | {q3[3]:^10.3f} |")
    print(f" {'Q2':^8} | {q2[0]:^11.3f} | {q2[1]:^11.3f} | {q2[2]:^10.3f} | {q2[3]:^10.3f} |")
    print(f" {'Q1':^8} | {q1[0]:^11.3f} | {q1[1]:^11.3f} | {q1[2]:^10.3f} | {q1[3]:^10.3f} |")
    print("-" * len(header))
    print("="*80)
