import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visuals
sns.set(style="whitegrid")

# Load the CSV file (replace with your actual path)
csv_file = 'runs/detect/train40/results.csv'  # ← Update this path if needed
df = pd.read_csv(csv_file)

# Print first few rows to understand structure
print("First few rows of data:")
print(df.head())

# Rename columns for readability (optional)
df.columns = [
    'epoch', 'time', 
    'train_box', 'train_cls', 'train_dfl',
    'precision', 'recall', 'map50', 'map50_95',
    'val_box', 'val_cls', 'val_dfl',
    'lr0', 'lr1', 'lr2'
]

# Plotting function
def plot_metric(x, y, title, ylabel):
    plt.figure(figsize=(10, 4))
    sns.lineplot(x=x, y=y, data=df)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.show()

# === 1. Loss Curves ===
plot_metric('epoch', 'train_box', 'Training Box Loss', 'Loss')
plot_metric('epoch', 'train_cls', 'Training Class Loss', 'Loss')
plot_metric('epoch', 'train_dfl', 'Training DFL Loss', 'Loss')

plot_metric('epoch', 'val_box', 'Validation Box Loss', 'Loss')
plot_metric('epoch', 'val_cls', 'Validation Class Loss', 'Loss')
plot_metric('epoch', 'val_dfl', 'Validation DFL Loss', 'Loss')

# === 2. Accuracy Metrics ===
plot_metric('epoch', 'map50', 'mAP@0.5', 'mAP')
plot_metric('epoch', 'map50_95', 'mAP@0.5:0.95', 'mAP')

# === 3. Precision & Recall ===
plot_metric('epoch', 'precision', 'Precision @0.5 IoU', 'Precision')
plot_metric('epoch', 'recall', 'Recall @0.5 IoU', 'Recall')

# === 4. Learning Rates ===
plot_metric('epoch', 'lr0', 'Learning Rate Group 0', 'LR')
plot_metric('epoch', 'lr1', 'Learning Rate Group 1', 'LR')
plot_metric('epoch', 'lr2', 'Learning Rate Group 2', 'LR')