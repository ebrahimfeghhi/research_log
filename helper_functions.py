# Model validation CTC losses and PER
import pandas as pd
import numpy as np
import pickle
import pickle
import pandas as pd

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