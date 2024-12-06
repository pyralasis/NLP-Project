import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import matplotlib.colors as mcolors

# these scripts were created with the aid of ChatGPT allowinfg us to quickly inspect our data  

def plot_class_distribution(dataset, id2label, tarin_or_test):
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
    plt.title(f"Class Distribution in {tarin_or_test} Dataset")
    plt.xticks(bar_positions, sorted_classes, rotation=45, ha="right")
    plt.tight_layout()
    plt.show()



def plot_combined_class_distribution(dataset, id2label, train_or_test):
    """
    Plots the combined class distribution in a dataset, grouping all opinion scores (pos, neu, neg)
    into single bars for B-X and I-X classes while keeping other categories separate.
    
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

    # Initialize data structures for combined classes
    combined_classes = ['B-holder', 'I-holder', 'B-targ', 'I-targ', 'B-exp', 'I-exp']
    combined_counts = {cls: 0 for cls in combined_classes}
    colors = {
        "B-holder": "#f6ec4b",  # Yellow
        "I-holder": "#f6ec4b80",  # Transparent Yellow
        "B-targ": "#a3fc65",  # Green
        "I-targ": "#a3fc6580",  # Transparent Green
        "B-exp": "#007fff",  # Blue for all opinions
        "I-exp": "#007fff80",  # Transparent Blue for all opinions
    }

    # Group B-exp and I-exp (combining all polarities)
    for label, name in id2label.items():
        if name.startswith("B-"):
            if name in ["B-exp-Neg", "B-exp-Neu", "B-exp-Pos"]:
                combined_counts["B-exp"] += label_counts.get(label, 0)
            else:
                combined_counts[name] += label_counts.get(label, 0)
        elif name.startswith("I-"):
            if name in ["I-exp-Neg", "I-exp-Neu", "I-exp-Pos"]:
                combined_counts["I-exp"] += label_counts.get(label, 0)
            else:
                combined_counts[name] += label_counts.get(label, 0)

    # Prepare data for plotting
    sorted_classes = list(combined_counts.keys())
    sorted_counts = list(combined_counts.values())
    bar_colors = [colors[cls] for cls in sorted_classes]

    # Plot combined class distribution
    plt.figure(figsize=(10, 6))
    bar_positions = np.arange(len(sorted_classes))
    plt.bar(bar_positions, sorted_counts, color=bar_colors, width=0.8, edgecolor="black")

    # Add labels on top of each bar
    for idx, count in enumerate(sorted_counts):
        plt.text(idx, count + 0.02 * max(sorted_counts), str(count), ha="center", va="bottom", fontsize=8)

    # Adjust axis
    plt.xlabel("Classes")
    plt.ylabel("Frequency")
    plt.title(f"Combined Class Distribution in {train_or_test} Dataset")
    plt.xticks(bar_positions, sorted_classes, rotation=45, ha="right")
    plt.tight_layout()
    plt.show()



import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import matplotlib.colors as mcolors


def plot_aggregated_class_distribution(dataset, id2label, train_or_test):
    """
    Plots the aggregated class distribution in a dataset, combining all B- and I- tags for each class 
    and merging all expression polarities into one 'exp' category.
    
    Args:
        dataset: A Hugging Face-style dataset with a "labels" field.
        id2label: A dictionary mapping class IDs to their corresponding class names.
        train_or_test: A string indicating whether the dataset is for training or testing.
    """
    # Flatten all labels from the dataset
    all_labels = []
    for sample in dataset:
        all_labels.extend(sample["labels"])

    # Filter out ignored labels (-100)
    valid_labels = [label for label in all_labels if label != -100]

    # Count occurrences of each label
    label_counts = Counter(valid_labels)

    # Initialize categories for aggregation
    aggregated_classes = ['holder', 'targ', 'exp']
    aggregated_counts = {cls: 0 for cls in aggregated_classes}
    colors = {
        "holder": "#f6ec4b",  # Yellow
        "targ": "#a3fc65",  # Green
        "exp": "#007fff",  # Blue for expressions
    }

    # Aggregate counts
    for label, name in id2label.items():
        if name.startswith("B-") or name.startswith("I-"):
            if "holder" in name:
                aggregated_counts["holder"] += label_counts.get(label, 0)
            elif "targ" in name:
                aggregated_counts["targ"] += label_counts.get(label, 0)
            elif "exp" in name:
                aggregated_counts["exp"] += label_counts.get(label, 0)

    # Prepare data for plotting
    sorted_classes = list(aggregated_counts.keys())
    sorted_counts = list(aggregated_counts.values())
    bar_colors = [colors[cls] for cls in sorted_classes]

    # Plot aggregated class distribution
    plt.figure(figsize=(8, 6))
    bar_positions = np.arange(len(sorted_classes))
    plt.bar(bar_positions, sorted_counts, color=bar_colors, width=0.6, edgecolor="black")

    # Add labels on top of each bar
    for idx, count in enumerate(sorted_counts):
        plt.text(idx, count + 0.02 * max(sorted_counts), str(count), ha="center", va="bottom", fontsize=10)

    # Adjust axis
    plt.xlabel("Classes")
    plt.ylabel("Frequency")
    plt.title(f"Aggregated Class Distribution in {train_or_test} Dataset")
    plt.xticks(bar_positions, sorted_classes, rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
