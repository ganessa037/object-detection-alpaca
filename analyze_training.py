import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the training results
results_df = pd.read_csv('runs/detect/train40/results.csv')

# Remove any leading/trailing spaces from column names
results_df.columns = results_df.columns.str.strip()

# Create figure with subplots - only 2 plots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('YOLOv8n Training Analysis', fontsize=12, fontweight='bold')

# 1. Precision over epochs
ax1 = axes[0]
ax1.plot(results_df['epoch'], results_df['metrics/precision(B)'], 'b-', linewidth=2, marker='o', markersize=3)
ax1.set_title('Precision (B) over Epochs', fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Precision')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)

# Highlight best precision
best_precision_idx = results_df['metrics/precision(B)'].idxmax()
best_precision_epoch = results_df.loc[best_precision_idx, 'epoch']
best_precision_value = results_df.loc[best_precision_idx, 'metrics/precision(B)']
ax1.plot(best_precision_epoch, best_precision_value, 'ro', markersize=8, label=f'Best Precision: {best_precision_value:.3f} (Epoch {best_precision_epoch})')
ax1.legend()

# Get best mAP50 for reference
best_map50_idx = results_df['metrics/mAP50(B)'].idxmax()
best_map50_epoch = results_df.loc[best_map50_idx, 'epoch']
best_map50_value = results_df.loc[best_map50_idx, 'metrics/mAP50(B)']

# 2. Combined metrics comparison
ax2 = axes[1]
ax2.plot(results_df['epoch'], results_df['metrics/precision(B)'], 'b-', linewidth=2, label='Precision', marker='o', markersize=2)
ax2.plot(results_df['epoch'], results_df['metrics/recall(B)'], 'r-', linewidth=2, label='Recall', marker='s', markersize=2)
ax2.plot(results_df['epoch'], results_df['metrics/mAP50(B)'], 'g-', linewidth=2, label='mAP@0.5', marker='^', markersize=2)
ax2.plot(results_df['epoch'], results_df['metrics/mAP50-95(B)'], 'm-', linewidth=2, label='mAP@0.5:0.95', marker='d', markersize=2)

ax2.set_title('All Key Metrics Comparison', fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Metric Value')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1)
ax2.legend()

# Add vertical line at best model epoch
ax2.axvline(x=best_map50_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Model (Epoch {best_map50_epoch})')

plt.tight_layout()
plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print analysis
print("=== YOLO Best Model Selection Analysis ===")
print(f"\n🎯 YOLO selects 'best.pt' based on HIGHEST mAP@0.5:")
print(f"   Best mAP@0.5: {best_map50_value:.3f} at Epoch {best_map50_epoch}")

print(f"\n📊 Metrics at Best Model Epoch ({best_map50_epoch}):")
best_row = results_df.loc[best_map50_idx]
print(f"   Precision: {best_row['metrics/precision(B)']:.3f}")
print(f"   Recall: {best_row['metrics/recall(B)']:.3f}")
print(f"   mAP@0.5: {best_row['metrics/mAP50(B)']:.3f}")
print(f"   mAP@0.5:0.95: {best_row['metrics/mAP50-95(B)']:.3f}")

print(f"\n🔍 Best Precision Analysis:")
print(f"   Highest Precision: {best_precision_value:.3f} at Epoch {best_precision_epoch}")
if best_precision_epoch != best_map50_epoch:
    print(f"   ⚠️  Note: Best precision epoch ({best_precision_epoch}) ≠ Best model epoch ({best_map50_epoch})")
    print(f"   This is normal - YOLO prioritizes mAP@0.5 over pure precision")
else:
    print(f"   ✅ Best precision and best model are from the same epoch!")

print(f"\n📈 Training Progress:")
print(f"   Initial Precision (Epoch 1): {results_df.loc[0, 'metrics/precision(B)']:.3f}")
print(f"   Final Precision (Epoch 40): {results_df.loc[39, 'metrics/precision(B)']:.3f}")
print(f"   Improvement: {(results_df.loc[39, 'metrics/precision(B)'] - results_df.loc[0, 'metrics/precision(B)']):.3f}")

print(f"\n🎖️ Why mAP@0.5 is used for 'best' model:")
print(f"   - mAP@0.5 considers both precision AND recall")
print(f"   - It's measured across all confidence thresholds")
print(f"   - Better represents overall detection performance")
print(f"   - Precision alone can be misleading (high precision, low recall)")

print(f"\n💡 Your model's confidence issue explanation:")
print(f"   - Your model has decent precision (~{best_row['metrics/precision(B)']:.1%})")
print(f"   - But predictions need low confidence threshold (0.1)")
print(f"   - This suggests the model is 'cautious' in its predictions")
print(f"   - More training data or epochs could improve confidence")
