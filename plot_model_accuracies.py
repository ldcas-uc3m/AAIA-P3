import os
import re
import matplotlib.pyplot as plt
import numpy as np

def extract_accuracy(file_path):
    """Extract accuracy value from a model evaluation file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Search for accuracy line in the evaluation results
            match = re.search(r'Accuracy: (\d+\.\d+)', content)
            if match:
                return float(match.group(1))
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return None

def gather_model_accuracies(base_dir):
    """Gather accuracies from all model evaluation files."""
    model_dirs = ['gemma', 'llama', 'mistral', 'qwen2.5', 'smollm']
    model_accuracies = []
    
    for model_dir in model_dirs:
        dir_path = os.path.join(base_dir, model_dir)
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            continue
            
        for filename in os.listdir(dir_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(dir_path, filename)
                accuracy = extract_accuracy(file_path)
                if accuracy is not None:
                    # Store model name and its accuracy
                    model_name = os.path.basename(filename)[:-4]  # Remove .txt extension
                    model_accuracies.append((model_name, accuracy))
    
    # Sort by accuracy (descending)
    model_accuracies.sort(key=lambda x: x[1], reverse=True)
    return model_accuracies

def plot_accuracies(model_accuracies):
    """Create a bar chart of model accuracies."""
    models, accuracies = zip(*model_accuracies)
    
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar chart
    y_pos = np.arange(len(models))
    bars = plt.barh(y_pos, accuracies, align='center', alpha=0.8)
    
    # Color bars by model family
    colors = {'llama': 'royalblue', 'mistral': 'forestgreen', 
              'qwen': 'firebrick', 'gemma': 'purple', 'smollm': 'orange'}
    
    for i, (model, _) in enumerate(model_accuracies):
        for family, color in colors.items():
            if family in model.lower():
                bars[i].set_color(color)
                break
    
    plt.yticks(y_pos, models, fontsize=10)
    
    # Add labels and title
    plt.xlabel('Accuracy', fontsize=12)
    plt.title('LLM Model Accuracy Comparison (Highest to Lowest)', fontsize=14)
    
    # Add grid lines
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add accuracy values as text labels
    for i, v in enumerate(accuracies):
        plt.text(v + 0.01, i, f'{v:.4f}', va='center')
    
    # Add legend for model families - moved to upper left to avoid obstruction
    legend_handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors.values()]
    plt.legend(legend_handles, colors.keys(), loc='upper left', bbox_to_anchor=(0, -0.05), 
               ncol=len(colors), frameon=False)
    
    # Adjust layout with more padding at bottom for legend
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Save the figure
    plt.savefig('model_accuracies.png', dpi=600, bbox_inches='tight')

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_accuracies = gather_model_accuracies(base_dir)
    
    print("Model Accuracies (sorted by accuracy):")
    print("="*50)
    for model, accuracy in model_accuracies:
        print(f"{model}: {accuracy:.4f}")
    
    plot_accuracies(model_accuracies)
    print("\nChart saved as 'model_accuracies.png'")

if __name__ == "__main__":
    main()
