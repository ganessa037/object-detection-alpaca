import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the training runs to compare with their YOLO versions
training_runs = {
    'train': 'YOLOv9t',
    'train2': 'YOLOv10n', 
    'train35': 'YOLOv11n',
    'train40': 'YOLOv8n'
}
colors = ['blue', 'red', 'green', 'orange']
models_data = {}

print("=== YOLO Models Comparison Analysis ===\n")

# Load data for each model
for run, yolo_version in training_runs.items():
    csv_path = f'runs/detect/{run}/results.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        models_data[yolo_version] = df
        print(f"✅ Loaded {yolo_version}: {len(df)} epochs")
    else:
        print(f"❌ Missing: {csv_path}")

if not models_data:
    print("No training data found!")
    exit()

print(f"\nComparing {len(models_data)} YOLO models...\n")

# Create mAP@0.5 comparison plot only
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
fig.suptitle('YOLO Models mAP@0.5 Comparison', fontsize=16, fontweight='bold')

# mAP@0.5 Comparison
for i, (yolo_version, df) in enumerate(models_data.items()):
    ax.plot(df['epoch'], df['metrics/mAP50(B)'], color=colors[i], 
             linewidth=3, label=f'{yolo_version}', marker='^', markersize=4)

ax.set_title('mAP@0.5 Comparison - YOLO Models', fontweight='bold', fontsize=14)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('mAP@0.5', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.set_ylim(0, 1)

# Highlight best performance for each model
for i, (yolo_version, df) in enumerate(models_data.items()):
    best_map50_idx = df['metrics/mAP50(B)'].idxmax()
    best_epoch = df.loc[best_map50_idx, 'epoch']
    best_value = df.loc[best_map50_idx, 'metrics/mAP50(B)']
    ax.plot(best_epoch, best_value, 'o', color=colors[i], markersize=8, 
            markeredgecolor='black', markeredgewidth=2)

plt.tight_layout()
plt.savefig('models_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Detailed analysis table
print("="*80)
print("📊 FINAL METRICS COMPARISON")
print("="*80)

comparison_data = []
for run, df in models_data.items():
    final_epoch = df.iloc[-1]
    best_map50_idx = df['metrics/mAP50(B)'].idxmax()
    best_row = df.iloc[best_map50_idx]
    
    comparison_data.append({
        'Model': run,
        'Total_Epochs': len(df),
        'Best_mAP50_Epoch': best_row['epoch'],
        'Best_mAP50': best_row['metrics/mAP50(B)'],
        'Precision_at_Best': best_row['metrics/precision(B)'],
        'Recall_at_Best': best_row['metrics/recall(B)'],
        'mAP50-95_at_Best': best_row['metrics/mAP50-95(B)'],
        'Final_Precision': final_epoch['metrics/precision(B)'],
        'Final_Recall': final_epoch['metrics/recall(B)'],
        'Final_mAP50': final_epoch['metrics/mAP50(B)'],
        'Final_mAP50-95': final_epoch['metrics/mAP50-95(B)']
    })

# Create comparison DataFrame
comp_df = pd.DataFrame(comparison_data)
comp_df = comp_df.round(3)

print("\n🏆 BEST MODEL PERFORMANCE (at best mAP@0.5 epoch):")
print("-" * 80)
print(f"{'Model':<10} {'Epochs':<8} {'Best_Epoch':<11} {'mAP@0.5':<9} {'Precision':<11} {'Recall':<8} {'mAP50-95':<10}")
print("-" * 80)
for _, row in comp_df.iterrows():
    print(f"{row['Model']:<10} {row['Total_Epochs']:<8} {row['Best_mAP50_Epoch']:<11} {row['Best_mAP50']:<9} {row['Precision_at_Best']:<11} {row['Recall_at_Best']:<8} {row['mAP50-95_at_Best']:<10}")

print("\n📈 FINAL EPOCH PERFORMANCE:")
print("-" * 80)
print(f"{'Model':<10} {'Final_mAP@0.5':<13} {'Final_Precision':<16} {'Final_Recall':<13} {'Final_mAP50-95':<15}")
print("-" * 80)
for _, row in comp_df.iterrows():
    print(f"{row['Model']:<10} {row['Final_mAP50']:<13} {row['Final_Precision']:<16} {row['Final_Recall']:<13} {row['Final_mAP50-95']:<15}")

# Find the best model
best_model_idx = comp_df['Best_mAP50'].idxmax()
best_model = comp_df.iloc[best_model_idx]

print("\n" + "="*80)
print("🥇 WINNER: BEST OVERALL MODEL")
print("="*80)
print(f"🏆 Best Model: {best_model['Model']}")
print(f"📊 Best mAP@0.5: {best_model['Best_mAP50']:.3f} (at epoch {int(best_model['Best_mAP50_Epoch'])})")
print(f"🎯 Precision: {best_model['Precision_at_Best']:.3f}")
print(f"🔍 Recall: {best_model['Recall_at_Best']:.3f}")
print(f"📈 mAP@0.5:0.95: {best_model['mAP50-95_at_Best']:.3f}")

print(f"\n💡 Model Selection Recommendation:")
# Map YOLO version back to training folder
yolo_to_folder = {v: k for k, v in training_runs.items()}
best_folder = yolo_to_folder[best_model['Model']]
print(f"   Use: runs/detect/{best_folder}/weights/best.pt")
print(f"   This model achieved the highest mAP@0.5 = {best_model['Best_mAP50']:.3f}")

# Check for training lengths and suggest improvements
print(f"\n🔍 Training Length Analysis:")
for _, row in comp_df.iterrows():
    epochs = row['Total_Epochs']
    best_epoch = row['Best_mAP50_Epoch']
    if best_epoch > epochs * 0.8:  # Best model was in last 20% of training
        print(f"   ⚠️  {row['Model']}: Best epoch {int(best_epoch)}/{epochs} - might benefit from more training")
    else:
        print(f"   ✅ {row['Model']}: Best epoch {int(best_epoch)}/{epochs} - training length was adequate")

print(f"\n📊 Performance Insights:")
precisions = comp_df['Best_mAP50'].values
if np.std(precisions) < 0.05:
    print(f"   📌 All models perform similarly (std: {np.std(precisions):.3f})")
    print(f"   Consider using the model with the highest final epoch performance")
else:
    print(f"   📌 Significant performance differences found (std: {np.std(precisions):.3f})")
    print(f"   Clear winner: {best_model['Model']}")

print(f"\n🚀 Next Steps:")
print(f"   1. Use the best model: runs/detect/{best_folder}/weights/best.pt")
print(f"   2. Test predictions with confidence threshold around 0.1-0.3")
print(f"   3. Consider ensemble methods if multiple models perform well")
