import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import matplotlib.colors as mcolors


def plot_class_distribution(dataset, id2label):
    """
    Plots the class distribution in a dataset with labels on each bar, no spacing
    between B-X and I-X bars, and distinct colors for B and I tags.
    
    Args:
        dataset: A Hugging Face-style dataset with a "labels" field.
        id2label: A dictionary mapping class IDs to their corresponding class names.
    """
    # Flatten all labels from the dataset
    all_labels = []
    for sample in dataset:
        all_labels.extend(sample["labels"])

    # Filter out ignored labels (-100)
    valid_labels = [label for label in all_labels if label != -100]

    # Count occurrences of each label
    label_counts = Counter(valid_labels)

    # Group B-X and I-X together
    sorted_classes = []
    sorted_counts = []
    colors = []

    # Define distinct colors for each category
    base_colors = {
        "holder": "#f6ec4b",  #yellow
        "targ": "#a3fc65",    #green
        "exp-Neg": "#ff0000", #red
        "exp-Neu": "#8321ff", #purple
        "exp-Pos": "#007fff"  #blue
    }

    for label, name in id2label.items():
        if name.startswith("B-"):
            base_name = name[2:]
            i_label = label + 1  # Assume B-X and I-X are consecutive in id2label
            
            # Append B-X and I-X separately
            sorted_classes.append(name)
            sorted_classes.append(id2label[i_label])
            sorted_counts.append(label_counts.get(label, 0))
            sorted_counts.append(label_counts.get(i_label, 0))
            
            # Add colors: darker for B and lighter for I
            base_color = base_colors.get(base_name, "#7f7f7f")  # Default gray
            colors.append(base_color)  # Darker for B
            lighter_color = mcolors.to_rgba(base_color, alpha=0.5)  # Lighter for I
            colors.append(lighter_color) # Same color but less for I 

    # Plot class distribution
    plt.figure(figsize=(12, 6))
    bar_positions = np.arange(len(sorted_classes))
    plt.bar(bar_positions, sorted_counts, color=colors, width=0.8, edgecolor="black")

    # Add labels on top of each bar
    for idx, count in enumerate(sorted_counts):
        plt.text(idx, count + 0.02 * max(sorted_counts), str(count), ha="center", va="bottom", fontsize=8)

    # Adjust axis
    plt.xlabel("Classes")
    plt.ylabel("Frequency")
    plt.title("Class Distribution in Dataset")
    plt.xticks(bar_positions, sorted_classes, rotation=45, ha="right")
    plt.tight_layout()
    plt.show()