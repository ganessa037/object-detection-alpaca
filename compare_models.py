import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the training runs to compare
training_runs = ['train', 'train2', 'train35', 'train40']
colors = ['blue', 'red', 'green', 'orange']
models_data = {}

print("=== YOLO Models Comparison Analysis ===\n")

# Load data for each model
for run in training_runs:
    csv_path = f'runs/detect/{run}/results.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        models_data[run] = df
        print(f"✅ Loaded {run}: {len(df)} epochs")
    else:
        print(f"❌ Missing: {csv_path}")

if not models_data:
    print("No training data found!")
    exit()

print(f"\nComparing {len(models_data)} models...\n")

# Create comprehensive comparison plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('YOLO Models Comparison - All Training Runs', fontsize=16, fontweight='bold')

# 1. Precision Comparison
ax1 = axes[0, 0]
for i, (run, df) in enumerate(models_data.items()):
    ax1.plot(df['epoch'], df['metrics/precision(B)'], color=colors[i], 
             linewidth=2, label=f'{run}', marker='o', markersize=2)
ax1.set_title('Precision (B) Comparison', fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Precision')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim(0, 1)

# 2. Recall Comparison
ax2 = axes[0, 1]
for i, (run, df) in enumerate(models_data.items()):
    ax2.plot(df['epoch'], df['metrics/recall(B)'], color=colors[i], 
             linewidth=2, label=f'{run}', marker='s', markersize=2)
ax2.set_title('Recall (B) Comparison', fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Recall')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_ylim(0, 1)

# 3. mAP@0.5 Comparison
ax3 = axes[0, 2]
for i, (run, df) in enumerate(models_data.items()):
    ax3.plot(df['epoch'], df['metrics/mAP50(B)'], color=colors[i], 
             linewidth=2, label=f'{run}', marker='^', markersize=2)
ax3.set_title('mAP@0.5 Comparison', fontweight='bold')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('mAP@0.5')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_ylim(0, 1)

# 4. mAP@0.5:0.95 Comparison
ax4 = axes[1, 0]
for i, (run, df) in enumerate(models_data.items()):
    ax4.plot(df['epoch'], df['metrics/mAP50-95(B)'], color=colors[i], 
             linewidth=2, label=f'{run}', marker='d', markersize=2)
ax4.set_title('mAP@0.5:0.95 Comparison', fontweight='bold')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('mAP@0.5:0.95')
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.set_ylim(0, 1)

# 5. Training Loss Comparison
ax5 = axes[1, 1]
for i, (run, df) in enumerate(models_data.items()):
    if 'train/box_loss' in df.columns:
        ax5.plot(df['epoch'], df['train/box_loss'], color=colors[i], 
                 linewidth=2, label=f'{run}', marker='o', markersize=2)
ax5.set_title('Training Box Loss', fontweight='bold')
ax5.set_xlabel('Epoch')
ax5.set_ylabel('Box Loss')
ax5.grid(True, alpha=0.3)
ax5.legend()

# 6. Validation Loss Comparison
ax6 = axes[1, 2]
for i, (run, df) in enumerate(models_data.items()):
    if 'val/box_loss' in df.columns:
        ax6.plot(df['epoch'], df['val/box_loss'], color=colors[i], 
                 linewidth=2, label=f'{run}', marker='s', markersize=2)
ax6.set_title('Validation Box Loss', fontweight='bold')
ax6.set_xlabel('Epoch')
ax6.set_ylabel('Box Loss')
ax6.grid(True, alpha=0.3)
ax6.legend()

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
print(f"   Use: runs/detect/{best_model['Model']}/weights/best.pt")
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
print(f"   1. Use the best model: runs/detect/{best_model['Model']}/weights/best.pt")
print(f"   2. Test predictions with confidence threshold around 0.1-0.3")
print(f"   3. Consider ensemble methods if multiple models perform well")
