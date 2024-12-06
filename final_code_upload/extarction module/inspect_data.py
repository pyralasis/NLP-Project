import json
import matplotlib.pyplot as plt

# inspect metrics over epochs

# Load the JSON data
with open("epoch_losses.json", "r") as file:
    data = json.load(file)

# Filter relevant data, keeping only the last entry for each epoch
epoch_data = {}
for entry in data:
    if "epoch" in entry and "eval_loss" in entry:
        epoch_data[entry["epoch"]] = {
            "eval_loss": entry["eval_loss"],
            "holder_f1": entry.get("eval_Holder F1 Total", 0),
            "target_f1": entry.get("eval_Target F1 Total", 0),
            "exp_f1": entry.get("eval_Exp. F1 Total", 0),
        }

# Extract data sorted by epoch
sorted_epochs = sorted(epoch_data.keys())
eval_losses = [epoch_data[epoch]["eval_loss"] for epoch in sorted_epochs]
holder_f1 = [epoch_data[epoch]["holder_f1"] for epoch in sorted_epochs]
target_f1 = [epoch_data[epoch]["target_f1"] for epoch in sorted_epochs]
exp_f1 = [epoch_data[epoch]["exp_f1"] for epoch in sorted_epochs]

# Plot evaluation losses over epochs
plt.figure(figsize=(10, 6))
plt.plot(sorted_epochs, eval_losses, marker='o', label="Evaluation Loss")
plt.title("Evaluation Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Evaluation Loss")
plt.grid(True)
plt.legend()
plt.show()

# Plot Total F1 scores for Holder, Target, and Expression over epochs
plt.figure(figsize=(10, 6))
plt.plot(sorted_epochs, holder_f1, marker='o', label="Holder F1")
plt.plot(sorted_epochs, target_f1, marker='o', label="Target F1")
plt.plot(sorted_epochs, exp_f1, marker='o', label="Expression F1")
plt.title("Total F1 Scores Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.grid(True)
plt.legend()
plt.show()


# Load the JSON data for metrics
with open("metrics.json", "r") as file:
    metrics_data = json.load(file)

# Expression metrics
expression_metrics = {
    "B-exp-Neg": metrics_data["eval_B-exp-Neg F1"],
    "I-exp-Neg": metrics_data["eval_I-exp-Neg F1"],
    "B-exp-Neu": metrics_data["eval_B-exp-Neu F1"],
    "I-exp-Neu": metrics_data["eval_I-exp-Neu F1"],
    "B-exp-Pos": metrics_data["eval_B-exp-Pos F1"],
    "I-exp-Pos": metrics_data["eval_I-exp-Pos F1"]
}

# Create bar graph for expression metrics
labels = list(expression_metrics.keys())
values = list(expression_metrics.values())
colors = [
    "red", "red",  # Negative
    "blue", "blue",  # Neutral
    "green", "green"  # Positive
]

plt.figure(figsize=(10, 6))
plt.bar(labels, values, color=colors)
plt.title("Expression Metrics (F1 Scores)")
plt.ylabel("F1 Score")
plt.xlabel("Expression Types")
plt.grid(axis="y")
plt.show()