import matplotlib.pyplot as plt
import numpy as np
import os
import re
import glob

def extract_metrics_from_file(filepath):
    """Extract accuracy and execution time from an evaluation file."""
    try:
        with open(filepath, 'r') as file:
            content = file.read()
            
            # Extract accuracy
            accuracy_match = re.search(r'Accuracy: ([0-9.]+)', content)
            accuracy = float(accuracy_match.group(1)) if accuracy_match else None
            
            # Extract execution time
            time_match = re.search(r'Total execution time: ([0-9.]+)', content)
            execution_time = float(time_match.group(1)) if time_match else None
            
            return accuracy, execution_time
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None, None

def plot_model_comparison(model_names, accuracies, execution_times, output_file='model_comparison.png', 
                          x_min=None, x_max=None, y_min=None, y_max=None):
    """
    Create a scatter plot of model accuracy vs. execution time.
    
    Parameters:
    model_names (list): Names of the language models
    accuracies (list): Accuracy scores for each model
    execution_times (list): Execution times in seconds for each model
    output_file (str): Filename to save the plot
    x_min (float): Optional manual minimum x-axis value
    x_max (float): Optional manual maximum x-axis value
    y_min (float): Optional manual minimum y-axis value
    y_max (float): Optional manual maximum y-axis value
    """
    # Create a larger plot
    plt.figure(figsize=(12, 8))
    # Create scatter plot with different markers for each model
    markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*']
    for i in range(len(model_names)):
        plt.scatter(execution_times[i], accuracies[i], s=150, marker=markers[i % len(markers)], 
                   label=model_names[i])

    # Add labels for each point with better positioning
    # Calculate x-range for proper offset scaling
    if x_min is None or x_max is None:
        x_min_auto, x_max_auto = min(execution_times), max(execution_times)
        x_range = max(1, x_max_auto - x_min_auto)  # Prevent division by zero
    else:
        x_range = max(1, x_max - x_min)
    x_offset = x_range * 0.01  # 1% of range
    
    # Add annotations with custom offsets to avoid overlap
    for i, model in enumerate(model_names):
        # Dynamic positioning based on point position
        plt.annotate(model, 
                     (execution_times[i], accuracies[i]),
                     textcoords="offset points", 
                     xytext=(15, 10),  # Increased horizontal offset
                     ha='left',
                     fontsize=11,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Add labels and title
    plt.xlabel('Execution Time (seconds)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Language Model Accuracy vs. Execution Time', fontsize=16)

    # Add grid and set axis limits
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set axis limits with manual override options
    # Set x-axis limits
    if x_min is not None and x_max is not None:
        plt.xlim(x_min, x_max)
    else:
        x_min_auto, x_max_auto = min(execution_times), max(execution_times)
        plt.xlim(x_min_auto - x_range * 0.05, x_max_auto + x_range * 0.25)
    
    # Set y-axis limits
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    else:
        y_min_auto = max(0, min(accuracies) - 0.05) if accuracies else 0.5
        y_max_auto = min(1.0, max(accuracies) + 0.05) if accuracies else 1.0
        plt.ylim(y_min_auto, y_max_auto)
    
    # Move legend to bottom right
    plt.legend(loc='lower right', fontsize=12)

    # Save the figure with higher DPI
    plt.tight_layout()
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    # Get all evaluation files
    eval_dir = os.path.join(os.getcwd(), "mistral")
    eval_files = glob.glob(os.path.join(eval_dir, "*.txt"))
    
    # Prepare data
    model_names = []
    accuracies = []
    execution_times = []
    
    for file_path in eval_files:
        # Extract model name from filename
        model_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Extract metrics
        accuracy, execution_time = extract_metrics_from_file(file_path)
        
        if accuracy is not None and execution_time is not None:
            model_names.append(model_name)
            accuracies.append(accuracy)
            execution_times.append(execution_time)
            print(f"Processed {model_name}: Accuracy={accuracy}, Time={execution_time}s")
        else:
            print(f"Skipping {model_name} due to missing data")
    
    # Generate the plot with manual axis limits
    output_file = "language_model_comparison.png"
    
    if model_names:
        # Set manual axis limits if needed
        x_min = 0  # Start time axis at 0 seconds
        x_max = None  # Auto-calculate max based on data
        y_min = 0.8  # Set minimum accuracy to 0.8 for better visibility
        y_max = 1.0  # Maximum accuracy is 1.0
        
        plot_model_comparison(model_names, accuracies, execution_times, output_file, 
                              x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    else:
        print("No valid data found in evaluation files.")
