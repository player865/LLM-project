import os.path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 配置输出目录
output_dir = "./output/role_multiround_experiment_final"
result_file = os.path.join(output_dir, "role_permutation_results.csv")

# 检查文件是否存在
if not os.path.exists(result_file):
    print(f"错误：结果文件不存在：{result_file}")
    print("请先运行 role_multiround_experiment.py 生成实验结果")
    exit(1)

# 读取数据
df = pd.read_csv(result_file)

# 1. 删除 output 和 target 为 NaN 的行
df = df.dropna(subset=['output', 'target'])

# 2. 处理 role_name 列：将 NaN 转换为"未知角色"
df['role_name'] = df['role_name'].fillna('未知角色')
df['role_name'] = df['role_name'].replace('', '无角色')

# 检查是否有 flag 字段
has_flag = 'flag' in df.columns

if has_flag:
    print(f"\n✅ 使用实验中已计算的 flag 字段进行判定（LLM 智能判定）")
    print(f"   flag=1 表示正确，flag=0 表示错误")
else:
    print(f"\n⚠️  未找到 flag 字段，将使用字符串完全匹配进行判定")
    print(f"   建议实验过程中保存 flag 字段以便更准确的评估")

print("=" * 80)
print("角色分配多轮实验 - 评估报告")
print("=" * 80)
print(f"\n数据概览:")
print(f"  总记录数：{len(df)}")
print(f"  唯一角色分配方案数：{df['role_assignment_id'].nunique()}")
print(f"  唯一模型数：{df['model_name'].nunique()}")
print(f"  角色分配方案: {df['role_assignment_id'].unique()}")

# 3. 创建全局类别映射（从 target 和 output 收集所有诊断类别）
all_categories = set()

for text in pd.concat([df['target'], df['output']]):
    all_categories.add(str(text).strip())

category_list = sorted(list(all_categories))
print(f"\n发现 {len(category_list)} 个唯一的诊断类别")

# 4. 创建全局标签编码器
label_encoder = LabelEncoder()
label_encoder.fit(category_list)

# 5. 定义一个函数来计算每个分组的指标
def calculate_metrics_for_group(group_df, label_encoder, group_name, use_flag=False):
    """为单个分组计算所有评估指标"""
    if len(group_df) == 0:
        return None

    group_df = group_df.copy()

    if use_flag and 'flag' in group_df.columns:
        # 使用 flag 字段（LLM 智能判定的结果）
        # flag=1 表示正确，flag=0 表示错误
        valid_mask = group_df['flag'] == 1
        n_samples = len(group_df)
        n_correct = valid_mask.sum()
        n_unknown = 0  # flag 已经包含了所有判定
        
        if not valid_mask.any():
            return None
        
        # 计算准确率
        accuracy = n_correct / n_samples
        
        # 对于二分类（正确/错误），其他指标简化计算
        # precision = TP / (TP + FP)
        # recall = TP / (TP + FN)
        # 这里 TP=n_correct, FP=0, FN=n_samples-n_correct
        tp = n_correct
        fp = 0
        fn = n_samples - n_correct
        tn = 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'n_samples': n_samples,
            'n_correct': n_correct,
            'n_unknown': n_unknown,
            'accuracy': accuracy,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1,
            'precision_micro': precision,
            'recall_micro': recall,
            'f1_micro': f1,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1
        }
    else:
        # 使用传统的字符串匹配方式
        # 将文本转换为数字标签
        group_df['y_true'] = label_encoder.transform(group_df['target'])

        # 处理 output 中的未知类别
        def encode_output(x):
            x_str = str(x).strip()
            if x_str in label_encoder.classes_:
                return label_encoder.transform([x_str])[0]
            else:
                return -1

        group_df['y_pred'] = group_df['output'].apply(encode_output)

        # 过滤掉预测为未知类别的样本
        valid_mask = group_df['y_pred'] != -1
        if not valid_mask.any():
            return None

        group_df_valid = group_df[valid_mask]
        y_true = group_df_valid['y_true'].values
        y_pred = group_df_valid['y_pred'].values

        n_samples = len(y_true)
        n_unknown = (~valid_mask).sum()

        try:
            # 基础指标
            accuracy = accuracy_score(y_true, y_pred)

            # 宏平均指标
            precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

            # 微平均指标
            precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
            recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
            f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)

            # 加权平均指标
            precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

            return {
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
                'f1_weighted': f1_weighted
            }
        except Exception as e:
            print(f"  计算指标时出错 - {e}")
            return None


# 6. 按 role_assignment_id 分组计算指标
print("\n" + "=" * 80)
print("按角色分配方案 (role_assignment_id) 评估指标")
print("=" * 80)

role_assignments = sorted(df['role_assignment_id'].unique())
results_by_role_assignment = {}

total_combinations = len(role_assignments)
current_combination = 0

for role_assignment_id in role_assignments:
    current_combination += 1
    print(f"\n[{current_combination}/{total_combinations}] 评估：{role_assignment_id}")

    combo_df = df[df['role_assignment_id'] == role_assignment_id]

    if len(combo_df) == 0:
        print(f"  -> 无数据，跳过")
        continue

    print(f"  -> 样本数：{len(combo_df)}")

    metrics = calculate_metrics_for_group(combo_df, label_encoder, role_assignment_id, use_flag=has_flag)

    if metrics is None:
        print(f"  -> 计算失败或无有效预测")
    else:
        results_by_role_assignment[role_assignment_id] = metrics
        if has_flag:
            print(f"  -> 准确率：{metrics['accuracy']:.4f} (基于 flag={metrics['n_correct']}/{metrics['n_samples']})")
        else:
            print(f"  -> 准确率：{metrics['accuracy']:.4f}, F1 宏平均：{metrics['f1_macro']:.4f}, F1 加权：{metrics['f1_weighted']:.4f}")

# 7. 按模型 + 角色分配方案分组计算指标
print("\n" + "=" * 80)
print("按模型 + 角色分配方案评估指标")
print("=" * 80)

models = sorted(df['model_name'].unique())
results_by_model_role = {}

for model in models:
    results_by_model_role[model] = {}
    for role_assignment_id in role_assignments:
        results_by_model_role[model][role_assignment_id] = None

total_combinations = len(models) * len(role_assignments)
current_combination = 0

for model in models:
    for role_assignment_id in role_assignments:
        current_combination += 1
        combo_df = df[(df['model_name'] == model) & (df['role_assignment_id'] == role_assignment_id)]

        if len(combo_df) == 0:
            continue

        metrics = calculate_metrics_for_group(combo_df, label_encoder, f"{model}-{role_assignment_id}")

        if metrics is not None:
            results_by_model_role[model][role_assignment_id] = metrics

# 8. 创建指标表格
print("\n" + "=" * 80)
print("指标表格")
print("=" * 80)

all_metrics = [
    'accuracy',
    'precision_macro', 'recall_macro', 'f1_macro',
    'precision_micro', 'recall_micro', 'f1_micro',
    'precision_weighted', 'recall_weighted', 'f1_weighted'
]

metric_names = {
    'accuracy': '准确率',
    'precision_macro': '精确率 (宏)',
    'recall_macro': '召回率 (宏)',
    'f1_macro': 'F1 分数 (宏)',
    'precision_micro': '精确率 (微)',
    'recall_micro': '召回率 (微)',
    'f1_micro': 'F1 分数 (微)',
    'precision_weighted': '精确率 (加权)',
    'recall_weighted': '召回率 (加权)',
    'f1_weighted': 'F1 分数 (加权)'
}

# 8.1 角色分配方案的综合指标表格
print("\n" + "-" * 80)
print("表 1: 每个角色分配方案的综合指标")
print("-" * 80)

for metric in all_metrics:
    print(f"\n{metric_names[metric]} ({metric}):")
    table = pd.DataFrame(index=role_assignments, columns=['Value'])
    for role_assignment_id in role_assignments:
        if role_assignment_id in results_by_role_assignment:
            table.loc[role_assignment_id, 'Value'] = results_by_role_assignment[role_assignment_id][metric]
        else:
            table.loc[role_assignment_id, 'Value'] = np.nan
    print(table.round(4))

# 8.2 模型在不同角色分配方案下的表现
print("\n" + "-" * 80)
print("表 2: 每个模型在不同角色分配方案下的准确率")
print("-" * 80)

accuracy_table = pd.DataFrame(index=models, columns=role_assignments)
for model in models:
    for role_assignment_id in role_assignments:
        if results_by_model_role[model][role_assignment_id] is not None:
            accuracy_table.loc[model, role_assignment_id] = results_by_model_role[model][role_assignment_id]['accuracy']
        else:
            accuracy_table.loc[model, role_assignment_id] = np.nan

print(accuracy_table.round(4))

# 9. 找出最佳角色分配方案
print("\n" + "=" * 80)
print("最佳角色分配方案分析")
print("=" * 80)

if results_by_role_assignment:
    # 按准确率排序
    sorted_by_accuracy = sorted(
        results_by_role_assignment.items(),
        key=lambda x: x[1]['accuracy'],
        reverse=True
    )

    print("\n角色分配方案按准确率排序:")
    for i, (role_id, metrics) in enumerate(sorted_by_accuracy, 1):
        print(f"  {i}. {role_id}")
        print(f"     准确率：{metrics['accuracy']:.4f}")
        print(f"     F1 加权：{metrics['f1_weighted']:.4f}")
        print(f"     样本数：{metrics['n_samples']}")

    best_role_id = sorted_by_accuracy[0][0]
    best_metrics = sorted_by_accuracy[0][1]
    print(f"\n🏆 最佳角色分配方案：{best_role_id}")
    print(f"   准确率：{best_metrics['accuracy']:.4f}")
    print(f"   F1 加权：{best_metrics['f1_weighted']:.4f}")

# 10. 找出每个模型的最佳角色分配方案
print("\n" + "=" * 80)
print("每个模型的最佳角色分配方案")
print("=" * 80)

for model in models:
    model_results = results_by_model_role[model]
    valid_results = {k: v for k, v in model_results.items() if v is not None}

    if valid_results:
        best_role_id = max(valid_results, key=lambda x: valid_results[x]['accuracy'])
        best_accuracy = valid_results[best_role_id]['accuracy']
        worst_role_id = min(valid_results, key=lambda x: valid_results[x]['accuracy'])
        worst_accuracy = valid_results[worst_role_id]['accuracy']
        avg_accuracy = np.mean([v['accuracy'] for v in valid_results.values()])

        print(f"\n模型：{model}")
        print(f"  最佳角色分配：{best_role_id} (准确率：{best_accuracy:.4f})")
        print(f"  最差角色分配：{worst_role_id} (准确率：{worst_accuracy:.4f})")
        print(f"  平均准确率：{avg_accuracy:.4f}")
        print(f"  准确率范围：{worst_accuracy:.4f} - {best_accuracy:.4f}")

# 11. 保存详细结果到 CSV
print("\n" + "=" * 80)
print("保存详细结果")
print("=" * 80)

detailed_results = []

for role_assignment_id, metrics in results_by_role_assignment.items():
    result_row = {
        'role_assignment_id': role_assignment_id,
        'n_samples': metrics['n_samples'],
        'n_unknown': metrics['n_unknown']
    }
    for metric in all_metrics:
        result_row[metric] = metrics[metric]
    detailed_results.append(result_row)

if detailed_results:
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df = detailed_df.sort_values('accuracy', ascending=False)

    output_file = os.path.join(output_dir, "role_assignment_all_metrics.csv")
    detailed_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存到：{output_file}")

# 12. 可视化展示
print("\n" + "=" * 80)
print("可视化展示")
print("=" * 80)

try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # 12.1 角色分配方案准确率对比条形图
    if results_by_role_assignment:
        role_ids = list(results_by_role_assignment.keys())
        accuracies = [results_by_role_assignment[r]['accuracy'] for r in role_ids]
        f1_weighted = [results_by_role_assignment[r]['f1_weighted'] for r in role_ids]

        x = np.arange(len(role_ids))
        width = 0.35

        bars1 = axes[0, 0].bar(x - width/2, accuracies, width, label='准确率', color='#1f77b4', alpha=0.8)
        bars2 = axes[0, 0].bar(x + width/2, f1_weighted, width, label='F1 加权', color='#ff7f0e', alpha=0.8)

        axes[0, 0].set_xlabel('角色分配方案')
        axes[0, 0].set_ylabel('指标值')
        axes[0, 0].set_title('不同角色分配方案的性能对比')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([f'Scenario {i+1}' for i in range(len(role_ids))], rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].set_ylim([0, 1])

        # 在条形上添加数值
        for bar in bars1:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    # 12.2 模型在不同角色下的准确率热力图
    if not accuracy_table.isnull().all().all():
        im2 = axes[0, 1].imshow(accuracy_table.astype(float), cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        axes[0, 1].set_title('每个模型在不同角色分配方案下的准确率热力图')
        axes[0, 1].set_xlabel('角色分配方案')
        axes[0, 1].set_ylabel('模型')
        axes[0, 1].set_xticks(range(len(role_assignments)))
        axes[0, 1].set_xticklabels([f'S{i+1}' for i in range(len(role_assignments))], rotation=45, ha='right')
        axes[0, 1].set_yticks(range(len(models)))
        axes[0, 1].set_yticklabels(models)

        plt.colorbar(im2, ax=axes[0, 1])

        for i in range(len(models)):
            for j in range(len(role_assignments)):
                if not pd.isna(accuracy_table.iloc[i, j]):
                    axes[0, 1].text(j, i, f'{accuracy_table.iloc[i, j]:.3f}',
                                   ha='center', va='center', color='black', fontsize=9)

    # 12.3 每个模型的平均性能对比
    if results_by_model_role:
        model_avg_accuracies = []
        model_avg_f1s = []

        for model in models:
            valid_metrics = [v for v in results_by_model_role[model].values() if v is not None]
            if valid_metrics:
                model_avg_accuracies.append(np.mean([m['accuracy'] for m in valid_metrics]))
                model_avg_f1s.append(np.mean([m['f1_weighted'] for m in valid_metrics]))
            else:
                model_avg_accuracies.append(np.nan)
                model_avg_f1s.append(np.nan)

        x = np.arange(len(models))
        width = 0.35

        bars1 = axes[1, 0].bar(x - width/2, model_avg_accuracies, width, label='平均准确率', color='#2ca02c', alpha=0.8)
        bars2 = axes[1, 0].bar(x + width/2, model_avg_f1s, width, label='平均 F1 加权', color='#d62728', alpha=0.8)

        axes[1, 0].set_xlabel('模型')
        axes[1, 0].set_ylabel('指标值')
        axes[1, 0].set_title('每个模型的平均性能对比')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].set_ylim([0, 1])

        for bar in bars1:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    # 12.4 样本数量分布
    if results_by_role_assignment:
        role_ids = list(results_by_role_assignment.keys())
        n_samples = [results_by_role_assignment[r]['n_samples'] for r in role_ids]

        axes[1, 1].bar(range(len(role_ids)), n_samples, color='#9467bd', alpha=0.8)
        axes[1, 1].set_xlabel('角色分配方案')
        axes[1, 1].set_ylabel('样本数量')
        axes[1, 1].set_title('不同角色分配方案的样本数量分布')
        axes[1, 1].set_xticks(range(len(role_ids)))
        axes[1, 1].set_xticklabels([f'Scenario {i+1}' for i in range(len(role_ids))], rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        for i, v in enumerate(n_samples):
            axes[1, 1].text(i, v, str(v), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    visualization_file = os.path.join(output_dir, 'role_assignment_metrics_visualization.png')
    plt.savefig(visualization_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"可视化图表已保存为：{visualization_file}")

except ImportError:
    print("\n注意：matplotlib 未安装，跳过可视化。如需可视化请运行：pip install matplotlib")
except Exception as e:
    print(f"\n可视化时出错：{e}")

print("\n" + "=" * 80)
print("评估完成!")
print("=" * 80)
