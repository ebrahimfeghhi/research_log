# Model validation CTC losses and PER
import pandas as pd
import numpy as np
import pickle
import pickle
import pandas as pd
from scipy.stats import ttest_rel


def process_and_display_results(pickle_paths, model_names=None):
    """
    Loads data from a list of pickle files, processes it, and prints a formatted Markdown table.

    Args:
        pickle_paths (list): A list of strings, where each string is a path to a pickle file.
        model_names (list, optional): A list of custom model names to use in the output table.
                                      If an element is None, the original model name from the pickle
                                      file is used for that entry.
    """
    all_data = []

    if model_names and len(model_names) != len(pickle_paths):
        raise ValueError("The number of custom model names must match the number of pickle paths.")

    for i, path in enumerate(pickle_paths):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            # Assuming each pickle file contains a single top-level dictionary
            model_key = list(data.keys())[0]
            metrics = data[model_key]

            # Use the custom name if provided and not None, otherwise use the original key.
            current_model_name = model_names[i] if model_names and model_names[i] is not None else model_key
            
            # Calculate the mean for each metric
            per_mean = sum(metrics['PER']) / len(metrics['PER'])
            wer_mean = sum(metrics['WER']) / len(metrics['WER'])
            ctc_loss_mean = sum(metrics['CTC Loss']) / len(metrics['CTC Loss'])
            
            # Append the processed data to the list
            all_data.append({
                'Model Name': current_model_name,
                'N': len(metrics['PER']), 
                'PER': f'{per_mean:.4f}',
                'CTC Loss': f'{ctc_loss_mean:.4f}',
                '3-gram WER': f'{wer_mean:.4f}'
            })

        except Exception as e:
            print(f"Error processing file {path}: {e}")

    if not all_data:
        print("No valid data to display.")
        return

    # Create and print the Markdown table
    df = pd.DataFrame(all_data)
    print(df.to_markdown(index=False))
    

def conduct_paired_ttest(model1_path, model2_path):
    
    model1_dict = pd.read_pickle(model1_path)
    model2_dict = pd.read_pickle(model2_path)
    
    model1_name = next(iter(model1_dict))
    model2_name = next(iter(model2_dict))
    
    model1_data = model1_dict[model1_name]
    model2_data = model2_dict[model2_name]
    
    stats_test_results = {}
    
    print(f"Stats test between {model1_name} and {model2_name}:\n")
    
    for key in model1_data.keys():
        
        m1_key_val = model1_data[key]
        m2_key_val = model2_data[key]
        
        r, p = ttest_rel(m1_key_val, m2_key_val)
        
        print(f"{key} - t value : {r: 0.4f}, p value: {p: 0.4f}")
        
        