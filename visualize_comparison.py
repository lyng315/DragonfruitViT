import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUTS_DIR = "outputs"

def plot_learning_curves(histories, metrics=['val_acc', 'val_loss', 'val_f1']):
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 5 * len(metrics)))
    if len(metrics) == 1: axes = [axes]
    
    for i, metric in enumerate(metrics):
        for model_name, df in histories.items():
            axes[i].plot(df['epoch'], df[metric], label=f'{model_name}')
        
        axes[i].set_title(f'Model Comparison: {metric.replace("_", " ").upper()}')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric.replace("_", " ").capitalize())
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'comparison_learning_curves.png'))
    plt.close()

def plot_individual_learning_curves(histories):
    for model_name, df in histories.items():
        if 'train_loss' in df.columns and 'val_loss' in df.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Subplot 1: Loss
            ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue', marker='o', markersize=4)
            ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', color='red', marker='x', markersize=4)
            ax1.set_title(f'[{model_name.upper()}] Training and Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Subplot 2: Accuracy
            ax2.plot(df['epoch'], df['train_acc'], label='Train Acc', color='blue', marker='o', markersize=4)
            ax2.plot(df['epoch'], df['val_acc'], label='Val Acc', color='red', marker='x', markersize=4)
            ax2.set_title(f'[{model_name.upper()}] Training and Validation Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUTS_DIR, f'learning_curve_{model_name}.png'))
            plt.close()

def plot_final_comparison(histories):
    results = []
    for model_name, df in histories.items():
        best_idx = df['val_acc'].idxmax()
        row = df.loc[best_idx].copy()
        row['model'] = model_name
        results.append(row)
    
    df_results = pd.DataFrame(results)
    compare_metrics = ['val_acc', 'val_precision', 'val_recall', 'val_f1']
    df_melted = df_results.melt(id_vars='model', value_vars=compare_metrics, 
                                var_name='Metric', value_name='Value')
    
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df_melted, x='Metric', y='Value', hue='model')
    plt.title('Benchmark: Final Evaluation Metrics (Best Epoch)')
    plt.ylim(0, 1.1)
    for p in plt.gca().patches:
        plt.gca().annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                           textcoords='offset points')
    
    plt.savefig(os.path.join(OUTPUTS_DIR, 'comparison_final_metrics.png'))
    plt.close()

def print_final_test_summary(models):
    print(f"\n{'='*25} FINAL TEST EVALUATION SUMMARY {'='*25}")
    print(f"{'Model':<15} | {'Test Acc':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 75)
    
    test_histories = []
    for model in models:
        file_path = os.path.join(OUTPUTS_DIR, f"test_results_{model}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            row = df.iloc[0]
            test_histories.append(row)
            print(f"{row['model']:<15} | {row['test_acc']:<10.4f} | {row['test_precision']:<10.4f} | {row['test_recall']:<10.4f} | {row['test_f1']:<10.4f}")
        else:
            print(f"{model:<15} | No test data available.")
            
    if test_histories:
        print("-" * 75)
        df_summary = pd.DataFrame(test_histories)
        df_summary = df_summary[['model', 'test_acc', 'test_precision', 'test_recall', 'test_f1']]
        df_summary.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
        df_summary.to_csv(os.path.join(OUTPUTS_DIR, 'model_comparison_table.csv'), index=False)

if __name__ == "__main__":
    models = ['vgg16', 'resnet50', 'mobilenetv2', 'vit']
    histories = {}
    
    for model in models:
        file_path = os.path.join(OUTPUTS_DIR, f"history_{model}.csv")
        if os.path.exists(file_path):
            histories[model] = pd.read_csv(file_path)
    
    if histories:
        plot_individual_learning_curves(histories)
        plot_learning_curves(histories)
        plot_final_comparison(histories)
        print_final_test_summary(models)
        print("\nVisualization pipeline completed.")
    else:
        print("Error: No training history found.")
