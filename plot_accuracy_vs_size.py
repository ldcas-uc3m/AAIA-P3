import matplotlib.pyplot as plt
import numpy as np
import os
import re
import glob

def extract_accuracy_from_file(filepath):
    """Extract accuracy from an evaluation file."""
    try:
        with open(filepath, 'r') as file:
            content = file.read()
            
            # Extract accuracy
            accuracy_match = re.search(r'Accuracy: ([0-9.]+)', content)
            accuracy = float(accuracy_match.group(1)) if accuracy_match else None
            
            return accuracy
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def plot_accuracy_vs_size(model_names, accuracies, sizes_gb, output_file='accuracy_vs_size.png',
                         y_min=None, y_max=None, x_min=None, x_max=None):
    """
    Create a scatter plot of model accuracy vs. size in GB.
    
    Parameters:
    model_names (list): Names of the language models
    accuracies (list): Accuracy scores for each model
    sizes_gb (list): Size in GB for each model
    output_file (str): Filename to save the plot
    y_min, y_max (float): Optional manual y-axis limits
    x_min, x_max (float): Optional manual x-axis limits
    """
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with different markers for each model
    markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*']
    for i in range(len(model_names)):
        plt.scatter(sizes_gb[i], accuracies[i], s=150, marker=markers[i % len(markers)], 
                   label=model_names[i])

    # Calculate axis ranges
    if x_min is None or x_max is None:
        x_min_auto, x_max_auto = min(sizes_gb), max(sizes_gb)
        x_range = max(1, x_max_auto - x_min_auto)
    else:
        x_range = max(1, x_max - x_min)
    
    # Add annotations with custom offsets to avoid overlap
    for i, model in enumerate(model_names):
        plt.annotate(model, 
                     (sizes_gb[i], accuracies[i]),
                     textcoords="offset points", 
                     xytext=(15, 10),
                     ha='left',
                     fontsize=11,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Add labels and title
    plt.xlabel('Model Size (GB)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Language Model Accuracy vs. Size', fontsize=16)

    # Add grid and set axis limits
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set axis limits
    if x_min is not None and x_max is not None:
        plt.xlim(x_min, x_max)
    else:
        plt.xlim(x_min_auto - x_range * 0.05, x_max_auto + x_range * 0.25)
    
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    else:
        y_min_auto = max(0, min(accuracies) - 0.05) if accuracies else 0.5
        y_max_auto = min(1.0, max(accuracies) + 0.05) if accuracies else 1.0
        plt.ylim(y_min_auto, y_max_auto)
    
    # Add legend
    plt.legend(loc='lower right', fontsize=12)

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    # Get the folder name to use in the output filename
    folder_name = "llama"
    
    # Define the output file path
    output_file = f"{folder_name}-accuracy-vs-size.png"
    
    # Get all evaluation files
    eval_dir = os.path.join(os.getcwd(), folder_name)
    eval_files = glob.glob(os.path.join(eval_dir, "*.txt"))
    
    # Define the model sizes in GB (this needs to be filled in manually)
    # The keys should match the filenames (without extension) in the evals folder
    model_sizes_gb = {
        "qwen2.5-0.5b-instruct-mlx@4bit": 0.29399,  # Converted MB to GB
        "qwen2.5-0.5b-instruct": 0.99961,           # Converted MB to GB
        "qwen2.5-1.5b-instruct": 3.10,
        "qwen2.5-3b-instruct-mlx@8bit": 3.30,
        "qwen2.5-7b-instruct-mlx@4bit": 4.30,
        "qwen2.5-3b-instruct": 6.18,
        "qwen2.5-7b-instruct-mlx@8bit": 8.11,
        "qwen2.5-14b-instruct-1m": 15.71,
        "qwen2.5-32b-instruct-mlx@8bit": 34.83,
        "qwen2.5-72b-instruct": 77.26,
        "qwen2.5-0.5b-instruct": 1.0,
        "qwen2.5-1.5b-instruct": 3.1,
        "qwen2.5-3b-instruct": 6.18,
        "qwen2.5-7b-instruct@4bit": 4.3,
        "qwen2.5-7b-instruct@8bit": 8.11,
        "qwen2.5-14b-instruct-1m@8bit": 15.7,
        "qwen2.5-32b-instruct@8bit": 34.8,
        "llama-3.2-1b-instruct": 2.48,
        "llama-3.1-8b-instruct@8bit": 8.54,
        "llama-3.1-8b-instruct": 16.08,
        "llama-3.2-3b-instruct": 6.43,
        "llama-3.3-70b-instruct@8bit": 74.98,
        "mistral-large-instruct-2407@4bit": 68.97,
        "mistral-small-24b-instruct-2501@8bit": 25.06
    }
    
    # Prepare data
    model_names = []
    accuracies = []
    sizes_gb = []
    
    for file_path in eval_files:
        # Extract model name from filename
        model_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Check if we have size information for this model
        if model_name not in model_sizes_gb:
            print(f"Skipping {model_name} - size information not available")
            continue
        
        # Extract accuracy
        accuracy = extract_accuracy_from_file(file_path)
        
        if accuracy is not None:
            model_names.append(model_name)
            accuracies.append(accuracy)
            sizes_gb.append(model_sizes_gb[model_name])
            print(f"Processed {model_name}: Accuracy={accuracy}, Size={model_sizes_gb[model_name]}GB")
        else:
            print(f"Skipping {model_name} due to missing accuracy data")
    
    # Generate the plot
    if model_names:
        # Set manual axis limits if needed
        y_min = None
        y_max = 1.0  # Maximum accuracy is 1.0
        x_min = 0    # Minimum size
        x_max = None # Auto-calculate based on data
        
        plot_accuracy_vs_size(model_names, accuracies, sizes_gb, output_file, 
                              y_min=y_min, y_max=y_max, x_min=x_min, x_max=x_max)
    else:
        print("No valid data found for plotting")
