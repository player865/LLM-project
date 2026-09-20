"""基于 flag 字段计算评估指标"""
import os
import pandas as pd
from glob import glob


def calculate_metrics_by_flag(csv_files):
    """根据 flag 字段计算指标"""
    results = []

    for csv_file in csv_files:
        print(f"\n{'='*60}")
        print(f"File: {csv_file}")
        print(f"{'='*60}")

        df = pd.read_csv(csv_file)

        if 'flag' not in df.columns:
            print(f"  Warning: 'flag' column not found in {csv_file}")
            continue

        model_name = df['model_name'].iloc[0] if 'model_name' in df.columns and len(df) > 0 else "unknown"
        assist_model = df['assist_model'].iloc[0] if 'assist_model' in df.columns and len(df) > 0 else "none"

        total = len(df)
        success = df[df['status'] == 'success']
        failed = df[df['status'] == 'failed']

        correct = df[df['flag'] == True]
        incorrect = df[df['flag'] == False]

        accuracy = len(correct) / total if total > 0 else 0
        error_rate = len(incorrect) / total if total > 0 else 0

        avg_rounds = df['conversation_rounds'].mean() if 'conversation_rounds' in df.columns else 0
        correct_rounds = correct['conversation_rounds'].mean() if len(correct) > 0 and 'conversation_rounds' in correct.columns else 0
        incorrect_rounds = incorrect['conversation_rounds'].mean() if len(incorrect) > 0 and 'conversation_rounds' in incorrect.columns else 0

        print(f"  Model: {model_name}")
        print(f"  Assist Model: {assist_model}")
        print(f"  Total samples: {total}")
        print(f"  Success: {len(success)}, Failed: {len(failed)}")
        print(f"  Correct: {len(correct)}, Incorrect: {len(incorrect)}")
        print(f"  Accuracy: {accuracy:.4f} ({len(correct)}/{total})")
        print(f"  Avg conversation rounds (all): {avg_rounds:.2f}")
        print(f"  Avg conversation rounds (correct): {correct_rounds:.2f}")
        print(f"  Avg conversation rounds (incorrect): {incorrect_rounds:.2f}")

        print(f"\n  --- Conversation Rounds Analysis ---")
        if 'conversation_rounds' in df.columns:
            rounds_analysis = df.groupby('conversation_rounds').agg({
                'flag': ['sum', 'count', 'mean']
            }).round(4)
            rounds_analysis.columns = ['correct', 'total', 'accuracy']
            rounds_analysis['accuracy'] = rounds_analysis['accuracy'].round(4)
            print(rounds_analysis.to_string())

        results.append({
            'model_name': model_name,
            'mode': assist_model,
            'n_samples': total,
            'n_unknown': len(failed),
            'accuracy': accuracy,
            'avg_conversation_rounds': avg_rounds,
            'avg_rounds_correct': correct_rounds,
            'avg_rounds_incorrect': incorrect_rounds
        })

    print(f"\n{'='*80}")
    print("Summary Table")
    print(f"{'='*80}")

    column_order = ['model_name', 'mode', 'n_samples', 'n_unknown', 'accuracy',
                   'avg_conversation_rounds', 'avg_rounds_correct', 'avg_rounds_incorrect']

    results_df = pd.DataFrame(results)
    results_df = results_df[column_order]
    print(results_df.to_string(index=False))

    return results_df


def analyze_by_assist_model(output_dir):
    """按 assist_model 分组分析"""
    print(f"\n{'='*80}")
    print("Analysis by Assist Model")
    print(f"{'='*80}")

    csv_files = glob(os.path.join(output_dir, "multiround_results*.csv"))
    csv_files = [f for f in csv_files if os.path.basename(f) != 'metrics_summary.csv']

    all_dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_dfs.append(df)

    if not all_dfs:
        print("No data found")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    if 'assist_model' in combined_df.columns:
        grouped = combined_df.groupby('assist_model').agg({
            'flag': ['sum', 'count', 'mean'],
            'conversation_rounds': 'mean'
        }).round(4)
        grouped.columns = ['correct', 'total', 'accuracy', 'avg_rounds']
        grouped['accuracy'] = grouped['accuracy'].round(4)
        print(grouped.to_string())
    else:
        print("assist_model column not found, showing overall stats:")
        acc = combined_df['flag'].mean()
        print(f"  Overall Accuracy: {acc:.4f}")


if __name__ == '__main__':
    output_dir = "./output/multiround_eval_deepseek_no_options"

    csv_files = glob(os.path.join(output_dir, "multiround_results*.csv"))
    csv_files = [f for f in csv_files if os.path.basename(f) != 'metrics_summary.csv']

    print(f"Found {len(csv_files)} result files:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")

    results_df = calculate_metrics_by_flag(csv_files)

    analyze_by_assist_model(output_dir)

    summary_file = os.path.join(output_dir, "metrics_summary.csv")
    results_df.to_csv(summary_file, index=False)
    print(f"\nSummary saved to: {summary_file}")
