"""
-*- coding: utf-8 -*-

@Author  : houcg
@Time    : 2026/1/26 10:30
"""
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# output_dir = "output/multiround_eval_with_options"
# result_file = os.path.join(output_dir, "multiround_results.csv")


output_dir = "output/multiround_eval_deepseek_no_options"
result_file = os.path.join(output_dir, "multiround_results_claude_opus_4_5_20251101.csv")



if not os.path.exists(result_file):
    print(f"Error: Results file not found {result_file}")
    print("Please run multiround_experiment.py first to generate results")
    exit(1)

df = pd.read_csv(result_file)

df = df.dropna(subset=['output', 'target'])

print("=" * 80)
print("Multi-Round Dialogue Test Results Analysis")
print("=" * 80)

print("\nRaw Data Overview:")
print(f"Total samples: {len(df)}")
print(f"Models: {df['model_name'].unique()}")
print(f"\nData columns: {df.columns.tolist()}")

has_options = 'use_options' in df.columns
if has_options:
    print(f"\nOptions mode distribution:")
    print(df['use_options'].value_counts())

print(f"\nStatus distribution:")
print(df['status'].value_counts())

print(f"\nPrediction correct/incorrect distribution:")
print(df['flag'].value_counts())

all_categories = set()
for text in pd.concat([df['target'], df['output']]):
    all_categories.add(str(text).strip())

category_list = sorted(list(all_categories))
print(f"\nFound {len(category_list)} unique diagnosis categories")

label_encoder = LabelEncoder()
label_encoder.fit(category_list)


def calculate_metrics_for_model(model_df, label_encoder, model_name, mode_label="all"):
    """Calculate evaluation metrics for a single model"""
    if len(model_df) == 0:
        return None

    model_df = model_df.copy()

    model_df['y_true'] = label_encoder.transform(model_df['target'])

    def encode_output(x):
        x_str = str(x).strip()
        if x_str in label_encoder.classes_:
            return label_encoder.transform([x_str])[0]
        else:
            return -1

    model_df['y_pred'] = model_df['output'].apply(encode_output)

    valid_mask = model_df['y_pred'] != -1
    if not valid_mask.any():
        print(f"  Model {model_name} ({mode_label}): No valid prediction samples")
        return None

    model_df_valid = model_df[valid_mask]
    y_true = model_df_valid['y_true'].values
    y_pred = model_df_valid['y_pred'].values

    n_samples = len(y_true)
    n_unknown = (~valid_mask).sum()

    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

        precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
        recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)

        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        avg_rounds = model_df['conversation_rounds'].mean() if 'conversation_rounds' in model_df.columns else 0

        correct_rounds = model_df[model_df['flag'] == True]['conversation_rounds'].mean() if len(model_df[model_df['flag'] == True]) > 0 else 0
        incorrect_rounds = model_df[model_df['flag'] == False]['conversation_rounds'].mean() if len(model_df[model_df['flag'] == False]) > 0 else 0

        return {
            'model_name': model_name,
            'mode': mode_label,
            'n_samples': n_samples,
            'n_unknown': n_unknown,
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'avg_conversation_rounds': avg_rounds,
            'avg_rounds_correct': correct_rounds,
            'avg_rounds_incorrect': incorrect_rounds
        }
    except Exception as e:
        print(f"  Model {model_name} ({mode_label}): Error calculating metrics - {e}")
        return None


print("\n" + "=" * 80)
print("Model Evaluation Metrics")
print("=" * 80)

results = []

if has_options:
    print("\n--- Analysis by Options Mode ---")

    with_options_df = df[df['use_options'] == True]
    without_options_df = df[df['use_options'] == False]

    print(f"\nWith options: {len(with_options_df)} samples")
    print(f"Without options: {len(without_options_df)} samples")

    if len(with_options_df) > 0:
        print("\n--- With Options Mode ---")
        grouped_with = with_options_df.groupby('model_name')
        for model_name, group in grouped_with:
            print(f"\n  Processing model: {model_name} (samples: {len(group)})")
            result = calculate_metrics_for_model(group, label_encoder, model_name, "with_options")
            if result:
                results.append(result)
                print(f"    Accuracy: {result['accuracy']:.4f}")
                print(f"    F1 Score (Macro): {result['f1_macro']:.4f}")
                print(f"    Avg conversation rounds: {result['avg_conversation_rounds']:.2f}")

    if len(without_options_df) > 0:
        print("\n--- Without Options Mode ---")
        grouped_without = without_options_df.groupby('model_name')
        for model_name, group in grouped_without:
            print(f"\n  Processing model: {model_name} (samples: {len(group)})")
            result = calculate_metrics_for_model(group, label_encoder, model_name, "without_options")
            if result:
                results.append(result)
                print(f"    Accuracy: {result['accuracy']:.4f}")
                print(f"    F1 Score (Macro): {result['f1_macro']:.4f}")
                print(f"    Avg conversation rounds: {result['avg_conversation_rounds']:.2f}")

    print("\n--- Combined Analysis (All Data) ---")
else:
    print("\n--- Analysis (No Options Mode Distinction) ---")
    if 'model_name' not in df.columns:
        print("Error: DataFrame has no 'model_name' column")
        result = calculate_metrics_for_model(df, label_encoder, "all_models", "all")
        if result:
            results.append(result)
    else:
        grouped = df.groupby('model_name')
        print(f"\nFound {len(grouped)} different models:")

        for model_name, group in grouped:
            print(f"\n  Processing model: {model_name} (samples: {len(group)})")
            result = calculate_metrics_for_model(group, label_encoder, model_name, "all")
            if result:
                results.append(result)
                print(f"    Accuracy: {result['accuracy']:.4f}")
                print(f"    F1 Score (Macro): {result['f1_macro']:.4f}")
                print(f"    Avg conversation rounds: {result['avg_conversation_rounds']:.2f}")

if results:
    results_df = pd.DataFrame(results)

    column_order = ['model_name', 'mode', 'n_samples', 'n_unknown', 'accuracy',
                    'precision_macro', 'recall_macro', 'f1_macro',
                    'precision_micro', 'recall_micro', 'f1_micro',
                    'precision_weighted', 'recall_weighted', 'f1_weighted',
                    'avg_conversation_rounds', 'avg_rounds_correct', 'avg_rounds_incorrect']
    results_df = results_df[column_order]

    print("\n" + "=" * 80)
    print("Detailed Evaluation Results Summary")
    print("=" * 80)

    for _, row in results_df.iterrows():
        print(f"\nModel: {row['model_name']} [{row['mode']}]")
        print(f"  Samples: {row['n_samples']} (unknown predictions: {row['n_unknown']})")
        print(f"  Accuracy: {row['accuracy']:.4f}")
        print(f"  Macro - Precision: {row['precision_macro']:.4f}, Recall: {row['recall_macro']:.4f}, F1: {row['f1_macro']:.4f}")
        print(f"  Micro - Precision: {row['precision_micro']:.4f}, Recall: {row['recall_micro']:.4f}, F1: {row['f1_micro']:.4f}")
        print(f"  Weighted - Precision: {row['precision_weighted']:.4f}, Recall: {row['recall_weighted']:.4f}, F1: {row['f1_weighted']:.4f}")
        if row['avg_conversation_rounds'] > 0:
            print(f"  Conversation rounds:")
            print(f"    Avg rounds: {row['avg_conversation_rounds']:.2f}")
            if row['avg_rounds_correct'] > 0:
                print(f"    Correct predictions avg rounds: {row['avg_rounds_correct']:.2f}")
            if row['avg_rounds_incorrect'] > 0:
                print(f"    Incorrect predictions avg rounds: {row['avg_rounds_incorrect']:.2f}")

    output_file = os.path.join(output_dir, "multiround_evaluation_deepseek_results.csv")
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nEvaluation results saved to: {output_file}")

    print("\n" + "=" * 80)
    print("Options Mode Comparison")
    print("=" * 80)

    if has_options:
        mode_comparison = results_df.pivot_table(
            values='accuracy',
            index='model_name',
            columns='mode',
            aggfunc='first'
        ).reset_index()

        if 'with_options' in mode_comparison.columns and 'without_options' in mode_comparison.columns:
            mode_comparison['improvement'] = mode_comparison['with_options'] - mode_comparison['without_options']
            print("\nAccuracy Comparison (With vs Without Options):")
            for _, row in mode_comparison.iterrows():
                model = row['model_name']
                with_acc = row.get('with_options', 0)
                without_acc = row.get('without_options', 0)
                improvement = row.get('improvement', 0)
                print(f"  {model}:")
                print(f"    With options: {with_acc:.4f}")
                print(f"    Without options: {without_acc:.4f}")
                print(f"    Improvement: {improvement:+.4f}")

    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    for mode in results_df['mode'].unique():
        mode_df = results_df[results_df['mode'] == mode]
        print(f"\n--- {mode.upper()} Mode ---")
        sorted_by_acc = mode_df.sort_values('accuracy', ascending=False)
        print("Sorted by Accuracy:")
        for i, (_, row) in enumerate(sorted_by_acc.iterrows()):
            print(f"  {i + 1}. {row['model_name']}: {row['accuracy']:.4f}")

        avg_accuracy = mode_df['accuracy'].mean()
        avg_f1_macro = mode_df['f1_macro'].mean()
        avg_rounds = mode_df['avg_conversation_rounds'].mean()

        print(f"\nAverage Metrics:")
        print(f"  Average Accuracy: {avg_accuracy:.4f}")
        print(f"  Average F1 Score (Macro): {avg_f1_macro:.4f}")
        print(f"  Average Conversation Rounds: {avg_rounds:.2f}")

    print("\n" + "=" * 80)
    print("Conversation Rounds vs Accuracy Analysis")
    print("=" * 80)

    if 'conversation_rounds' in df.columns:
        rounds_analysis = df.groupby('conversation_rounds').agg({
            'flag': ['sum', 'count', 'mean']
        }).reset_index()
        rounds_analysis.columns = ['Rounds', 'Correct', 'Total', 'Accuracy']
        rounds_analysis['Accuracy'] = rounds_analysis['Accuracy'].apply(lambda x: f"{x:.4f}")
        print("\nConversation Rounds Distribution:")
        print(rounds_analysis.to_string(index=False))

    print("\n" + "=" * 80)
    print("Self-Reflection Effect Analysis")
    print("=" * 80)

    success_df = df[df['status'] == 'success']
    if not success_df.empty:
        print(f"\nConversation Rounds Distribution:")
        rounds_dist = success_df['conversation_rounds'].value_counts().sort_index()
        for rounds, count in rounds_dist.items():
            pct = count / len(success_df) * 100
            print(f"  {rounds} rounds: {count} samples ({pct:.1f}%)")

        print(f"\nSelf-Reflection Effect Analysis:")
        correct_df = success_df[success_df['flag'] == True]
        incorrect_df = success_df[success_df['flag'] == False]

        if not correct_df.empty:
            print(f"  Average rounds for correct predictions: {correct_df['conversation_rounds'].mean():.2f}")
        if not incorrect_df.empty:
            print(f"  Average rounds for incorrect predictions: {incorrect_df['conversation_rounds'].mean():.2f}")

        if not correct_df.empty and not incorrect_df.empty:
            diff = incorrect_df['conversation_rounds'].mean() - correct_df['conversation_rounds'].mean()
            if diff > 0:
                print(f"  Incorrect predictions took more rounds ({diff:.2f}), indicating model may need more reflection but still fails")
            else:
                print(f"  Correct predictions took more rounds ({abs(diff):.2f}), indicating sufficient self-dialogue helps improve accuracy")

    print("\n" + "=" * 80)
    print("Visualization")
    print("=" * 80)

    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        if has_options:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        modes = results_df['mode'].unique()
        colors = {'claude': '#2ecc71', 'gpt': '#3498db', 'gemini': '#9b59b6'}

        if has_options:
            plot_idx = 0
            for mode in ['with_options', 'without_options']:
                mode_df = results_df[results_df['mode'] == mode]
                if len(mode_df) == 0:
                    continue

                models = mode_df['model_name']
                x = range(len(models))
                model_short = [m.split('-')[0] for m in models]

                axes[0, plot_idx].bar(x, mode_df['accuracy'], color=[colors.get(m.split('-')[0], '#95a5a6') for m in models])
                axes[0, plot_idx].set_title(f'Model Accuracy ({mode.replace("_", " ").title()})', fontsize=12)
                axes[0, plot_idx].set_xlabel('Model')
                axes[0, plot_idx].set_ylabel('Accuracy')
                axes[0, plot_idx].set_xticks(x)
                axes[0, plot_idx].set_xticklabels(model_short, rotation=45, ha='right')
                for i, v in enumerate(mode_df['accuracy']):
                    axes[0, plot_idx].text(i, v + 0.01, f'{v:.2%}', ha='center', fontsize=9)

                axes[1, plot_idx].bar(x, mode_df['f1_macro'], color=[colors.get(m.split('-')[0], '#95a5a6') for m in models])
                axes[1, plot_idx].set_title(f'Model F1 Score ({mode.replace("_", " ").title()})', fontsize=12)
                axes[1, plot_idx].set_xlabel('Model')
                axes[1, plot_idx].set_ylabel('F1 Score')
                axes[1, plot_idx].set_xticks(x)
                axes[1, plot_idx].set_xticklabels(model_short, rotation=45, ha='right')
                for i, v in enumerate(mode_df['f1_macro']):
                    axes[1, plot_idx].text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=9)

                plot_idx += 1

            ax_combined = axes[0, 2] if has_options else axes[1, 1]
            width = 0.35
            all_models = results_df['model_name'].unique()

            with_acc = results_df[results_df['mode'] == 'with_options'].set_index('model_name')['accuracy']
            without_acc = results_df[results_df['mode'] == 'without_options'].set_index('model_name')['accuracy']

            x_all = range(len(all_models))
            with_vals = [with_acc.get(m, 0) for m in all_models]
            without_vals = [without_acc.get(m, 0) for m in all_models]

            ax_combined.bar([i - width / 2 for i in x_all], with_vals, width, label='With Options', color='#3498db')
            ax_combined.bar([i + width / 2 for i in x_all], without_vals, width, label='Without Options', color='#e74c3c')
            ax_combined.set_title('Options Mode Comparison', fontsize=12)
            ax_combined.set_xlabel('Model')
            ax_combined.set_ylabel('Accuracy')
            ax_combined.set_xticks(x_all)
            ax_combined.set_xticklabels([m.split('-')[0] for m in all_models], rotation=45, ha='right')
            ax_combined.legend()

            rounds_ax = axes[1, 2] if has_options else None
            if rounds_ax:
                rounds_data = results_df[results_df['mode'] == 'with_options'][['model_name', 'avg_conversation_rounds']].copy()
                rounds_ax.bar(range(len(rounds_data)), rounds_data['avg_conversation_rounds'], color='#9b59b6')
                rounds_ax.set_title('Avg Conversation Rounds (With Options)', fontsize=12)
                rounds_ax.set_xlabel('Model')
                rounds_ax.set_ylabel('Rounds')
                rounds_ax.set_xticks(range(len(rounds_data)))
                rounds_ax.set_xticklabels([m.split('-')[0] for m in rounds_data['model_name']], rotation=45, ha='right')

        else:
            models = results_df['model_name']
            x = range(len(models))
            model_short = [m.split('-')[0] for m in models]

            axes[0, 0].bar(x, results_df['accuracy'], color=['#2ecc71', '#3498db', '#9b59b6'])
            axes[0, 0].set_title('Model Accuracy Comparison', fontsize=14)
            axes[0, 0].set_xlabel('Model')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(model_short, rotation=45, ha='right')
            for i, v in enumerate(results_df['accuracy']):
                axes[0, 0].text(i, v + 0.01, f'{v:.2%}', ha='center', fontsize=10)

            axes[0, 1].bar(x, results_df['f1_macro'], color=['#e74c3c', '#f39c12', '#1abc9c'])
            axes[0, 1].set_title('Model F1 Score (Macro) Comparison', fontsize=14)
            axes[0, 1].set_xlabel('Model')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(model_short, rotation=45, ha='right')
            for i, v in enumerate(results_df['f1_macro']):
                axes[0, 1].text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)

            width = 0.35
            axes[1, 0].bar([i - width / 2 for i in x], results_df['precision_macro'], width, label='Precision', color='#3498db')
            axes[1, 0].bar([i + width / 2 for i in x], results_df['recall_macro'], width, label='Recall', color='#e74c3c')
            axes[1, 0].set_title('Model Precision and Recall (Macro) Comparison', fontsize=14)
            axes[1, 0].set_xlabel('Model')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(model_short, rotation=45, ha='right')
            axes[1, 0].legend()

            rounds_data = results_df[['model_name', 'avg_conversation_rounds', 'avg_rounds_correct', 'avg_rounds_incorrect']].copy()
            axes[1, 1].bar(x, rounds_data['avg_conversation_rounds'], label='Avg Rounds', color='#9b59b6', alpha=0.7)
            axes[1, 1].bar(x, rounds_data['avg_rounds_correct'], label='Correct Rounds', color='#2ecc71', alpha=0.7)
            axes[1, 1].bar(x, rounds_data['avg_rounds_incorrect'], label='Incorrect Rounds', color='#e74c3c', alpha=0.7)
            axes[1, 1].set_title('Model Conversation Rounds Analysis', fontsize=14)
            axes[1, 1].set_xlabel('Model')
            axes[1, 1].set_ylabel('Rounds')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(model_short, rotation=45, ha='right')
            axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'multiround_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nVisualization saved to: {os.path.join(output_dir, 'multiround_comparison.png')}")

    except ImportError:
        print("\nNote: matplotlib not installed, skipping visualization. Run: pip install matplotlib")
    except Exception as e:
        print(f"\nVisualization error: {e}")

    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print(f"\nResult Files:")
    print(f"  1. {result_file}")
    print(f"  2. {output_file}")
    print(f"  3. {os.path.join(output_dir, 'multiround_comparison.png')}")

else:
    print("\nNo evaluation results available")
