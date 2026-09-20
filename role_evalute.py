import os.path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

output_dir = "./output/role_eval_new"
df = pd.read_csv(os.path.join(output_dir, "model_results.csv"))

# 1. 首先删除 output 和 target 为 NaN 的行
df = df.dropna(subset=['output', 'target'])

# 2. 处理role列：将NaN转换为"通用角色"
df['role'] = df['role'].fillna('通用角色')
# 同时处理空字符串角色
df['role'] = df['role'].replace('', '无角色')

print("原始数据示例:")
print(df[['role', 'model_name', 'output', 'target']].head())
print(f"\n总共有 {len(df)} 条数据")
print(f"角色: {df['role'].unique()}")
print(f"模型名称: {df['model_name'].unique()}")

# 3. 创建全局类别映射
# 获取所有唯一的诊断名称（来自target和output）
all_categories = set()

# 从target和output中收集所有可能的诊断
for text in pd.concat([df['target'], df['output']]):
    all_categories.add(str(text).strip())

# 将类别转换为列表并排序
category_list = sorted(list(all_categories))
print(f"\n发现 {len(category_list)} 个唯一的诊断类别")

# 4. 创建全局标签编码器
label_encoder = LabelEncoder()
label_encoder.fit(category_list)


# 5. 定义一个函数来计算每个分组的指标
def calculate_metrics_for_group(group_df, label_encoder, group_name):
    """为单个分组计算所有评估指标"""
    if len(group_df) == 0:
        return None

    # 复制数据以避免修改原始数据
    group_df = group_df.copy()

    # 将文本转换为数字标签
    group_df['y_true'] = label_encoder.transform(group_df['target'])

    # 处理output中的未知类别
    def encode_output(x):
        x_str = str(x).strip()
        if x_str in label_encoder.classes_:
            return label_encoder.transform([x_str])[0]
        else:
            # 如果不在已知类别中，返回-1
            return -1

    group_df['y_pred'] = group_df['output'].apply(encode_output)

    # 过滤掉预测为未知类别的样本
    valid_mask = group_df['y_pred'] != -1
    if not valid_mask.any():
        return None

    group_df_valid = group_df[valid_mask]
    y_true = group_df_valid['y_true'].values
    y_pred = group_df_valid['y_pred'].values

    # 计算样本数量
    n_samples = len(y_true)
    n_unknown = (~valid_mask).sum()

    # 计算所有评估指标
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


# 6. 计算每个模型在每个角色上的指标
print("\n" + "=" * 80)
print("计算每个模型在每个角色上的所有评估指标")
print("=" * 80)

# 获取所有模型和角色
models = sorted(df['model_name'].unique())
roles = sorted(df['role'].unique())

# 创建结果存储结构
results_by_model_role = {}

# 初始化结果字典
for model in models:
    results_by_model_role[model] = {}
    for role in roles:
        results_by_model_role[model][role] = None

# 计算每个模型-角色组合的指标
total_combinations = len(models) * len(roles)
current_combination = 0

print(f"共有 {len(models)} 个模型和 {len(roles)} 个角色，总计 {total_combinations} 个组合需要评估")

for model in models:
    for role in roles:
        current_combination += 1
        print(f"\n[{current_combination}/{total_combinations}] 评估组合: 模型={model}, 角色={role}")

        # 筛选该模型-角色组合的数据
        combo_df = df[(df['model_name'] == model) & (df['role'] == role)]

        if len(combo_df) == 0:
            print(f"  -> 无数据，跳过")
            continue

        print(f"  -> 样本数: {len(combo_df)}")

        # 计算所有指标
        metrics = calculate_metrics_for_group(combo_df, label_encoder, f"{model}-{role}")

        if metrics is None:
            print(f"  -> 计算失败或无有效预测")
        else:
            results_by_model_role[model][role] = metrics
            print(
                f"  -> 准确率: {metrics['accuracy']:.4f}, F1宏平均: {metrics['f1_macro']:.4f}, F1加权: {metrics['f1_weighted']:.4f}")

# 7. 将所有指标整理为表格形式
print("\n" + "=" * 80)
print("每个模型在每个角色上的所有评估指标表格")
print("=" * 80)

# 定义所有要计算的指标
all_metrics = [
    'accuracy',
    'precision_macro', 'recall_macro', 'f1_macro',
    'precision_micro', 'recall_micro', 'f1_micro',
    'precision_weighted', 'recall_weighted', 'f1_weighted'
]

metric_names = {
    'accuracy': '准确率',
    'precision_macro': '精确率(宏)',
    'recall_macro': '召回率(宏)',
    'f1_macro': 'F1分数(宏)',
    'precision_micro': '精确率(微)',
    'recall_micro': '召回率(微)',
    'f1_micro': 'F1分数(微)',
    'precision_weighted': '精确率(加权)',
    'recall_weighted': '召回率(加权)',
    'f1_weighted': 'F1分数(加权)'
}

# 为每个指标创建透视表
metric_tables = {}

for metric in all_metrics:
    print(f"\n{metric_names[metric]} ({metric}) 表格:")
    print("-" * 80)

    # 创建透视表
    metric_table = pd.DataFrame(index=models, columns=roles)
    for model in models:
        for role in roles:
            if results_by_model_role[model][role] is not None:
                metric_table.loc[model, role] = results_by_model_role[model][role][metric]
            else:
                metric_table.loc[model, role] = np.nan

    # 保存到字典
    metric_tables[metric] = metric_table

    # 显示表格
    print(metric_table.round(4))

# 8. 显示样本数量表格
print("\n\n样本数量表格:")
print("-" * 80)
samples_table = pd.DataFrame(index=models, columns=roles)
for model in models:
    for role in roles:
        if results_by_model_role[model][role] is not None:
            samples_table.loc[model, role] = results_by_model_role[model][role]['n_samples']
        else:
            samples_table.loc[model, role] = 0

print(samples_table)

# 9. 找出每个模型的最佳角色（基于准确率）
print("\n" + "=" * 80)
print("每个模型的最佳角色分析（基于准确率）")
print("=" * 80)

for model in models:
    # 获取该模型在所有角色上的准确率
    model_accuracies = {}
    for role in roles:
        if results_by_model_role[model][role] is not None:
            model_accuracies[role] = results_by_model_role[model][role]['accuracy']

    if model_accuracies:
        # 找出最佳准确率
        best_role = max(model_accuracies, key=model_accuracies.get)
        best_accuracy = model_accuracies[best_role]

        # 找出最差准确率
        worst_role = min(model_accuracies, key=model_accuracies.get)
        worst_accuracy = model_accuracies[worst_role]

        # 计算平均准确率
        avg_accuracy = np.mean(list(model_accuracies.values()))

        print(f"\n模型: {model}")
        print(f"  最佳角色: {best_role} (准确率: {best_accuracy:.4f})")
        print(f"  最差角色: {worst_role} (准确率: {worst_accuracy:.4f})")
        print(f"  平均准确率: {avg_accuracy:.4f}")
        print(f"  准确率范围: {worst_accuracy:.4f} - {best_accuracy:.4f}")

        # 按准确率排序显示所有角色
        print(f"  所有角色准确率排序:")
        sorted_roles = sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True)
        for i, (role, acc) in enumerate(sorted_roles):
            print(f"    {i + 1}. {role}: {acc:.4f}")

# 10. 找出每个模型的最佳角色（基于F1加权平均）
print("\n" + "=" * 80)
print("每个模型的最佳角色分析（基于F1加权平均）")
print("=" * 80)

for model in models:
    # 获取该模型在所有角色上的F1加权平均
    model_f1_weighted = {}
    for role in roles:
        if results_by_model_role[model][role] is not None:
            model_f1_weighted[role] = results_by_model_role[model][role]['f1_weighted']

    if model_f1_weighted:
        # 找出最佳F1加权平均
        best_role_f1 = max(model_f1_weighted, key=model_f1_weighted.get)
        best_f1 = model_f1_weighted[best_role_f1]

        print(f"\n模型: {model}")
        print(f"  最佳角色(F1加权): {best_role_f1} (F1加权: {best_f1:.4f})")

# 11. 保存详细结果到CSV
print("\n" + "=" * 80)
print("保存详细结果")
print("=" * 80)

# 准备详细结果数据
detailed_results = []

for model in models:
    for role in roles:
        if results_by_model_role[model][role] is not None:
            metrics = results_by_model_role[model][role]
            result_row = {
                'model': model,
                'role': role,
                'n_samples': metrics['n_samples'],
                'n_unknown': metrics['n_unknown']
            }
            # 添加所有评估指标
            for metric in all_metrics:
                result_row[metric] = metrics[metric]

            detailed_results.append(result_row)

# 转换为DataFrame并保存
if detailed_results:
    detailed_df = pd.DataFrame(detailed_results)

    # 按模型和角色排序
    detailed_df = detailed_df.sort_values(['model', 'role'])

    # 保存到CSV
    output_file = os.path.join(output_dir, "model_role_all_metrics.csv")
    detailed_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存到: {output_file}")
else:
    print("\n没有可用的评估结果")

# 12. 可视化展示
print("\n" + "=" * 80)
print("可视化展示")
print("=" * 80)

try:
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 选择几个关键指标进行可视化
    key_metrics = ['accuracy', 'f1_macro', 'f1_weighted']
    key_metric_names = ['准确率', 'F1分数(宏平均)', 'F1分数(加权平均)']

    # 创建可视化图表
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # 12.1 准确率热力图
    if not metric_tables['accuracy'].isnull().all().all():
        im1 = axes[0, 0].imshow(metric_tables['accuracy'].astype(float), cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        axes[0, 0].set_title('每个模型在每个角色上的准确率热力图')
        axes[0, 0].set_xlabel('角色')
        axes[0, 0].set_ylabel('模型')
        axes[0, 0].set_xticks(range(len(roles)))
        axes[0, 0].set_xticklabels(roles, rotation=45, ha='right')
        axes[0, 0].set_yticks(range(len(models)))
        axes[0, 0].set_yticklabels(models)

        # 添加颜色条
        plt.colorbar(im1, ax=axes[0, 0])

        # 在每个单元格中添加数值
        for i in range(len(models)):
            for j in range(len(roles)):
                if not pd.isna(metric_tables['accuracy'].iloc[i, j]):
                    axes[0, 0].text(j, i, f'{metric_tables["accuracy"].iloc[i, j]:.2f}',
                                    ha='center', va='center', color='black', fontsize=8)

    # 12.2 关键指标对比条形图
    if not metric_tables['accuracy'].isnull().all().all():
        x = np.arange(len(models))
        width = 0.25  # 每个条形的宽度
        metric_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 不同指标的颜色

        for idx, (metric, color) in enumerate(zip(key_metrics, metric_colors)):
            # 计算每个模型在所有角色上的平均指标
            avg_metrics = []
            for model in models:
                model_metrics = []
                for role in roles:
                    if not pd.isna(metric_tables[metric].loc[model, role]):
                        model_metrics.append(metric_tables[metric].loc[model, role])
                if model_metrics:
                    avg_metrics.append(np.mean(model_metrics))
                else:
                    avg_metrics.append(np.nan)

            axes[0, 1].bar(x + (idx - 1) * width, avg_metrics, width,
                           label=key_metric_names[idx], color=color, alpha=0.7)

        axes[0, 1].set_title('每个模型的关键指标平均值对比')
        axes[0, 1].set_xlabel('模型')
        axes[0, 1].set_ylabel('指标值')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].set_ylim([0, 1])

    # 12.3 每个模型的指标变化雷达图（简化版）
    if detailed_results:
        # 选择前3个模型进行展示
        top_models = models[:3] if len(models) >= 3 else models

        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(key_metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形

        ax = axes[1, 0]
        ax = plt.subplot(2, 2, 3, polar=True)

        for model_idx, model in enumerate(top_models):
            # 获取该模型在所有角色上的平均指标
            model_avg_metrics = []
            for metric in key_metrics:
                model_metrics = []
                for role in roles:
                    if not pd.isna(metric_tables[metric].loc[model, role]):
                        model_metrics.append(metric_tables[metric].loc[model, role])
                if model_metrics:
                    model_avg_metrics.append(np.mean(model_metrics))
                else:
                    model_avg_metrics.append(0)

            # 闭合图形
            model_avg_metrics += model_avg_metrics[:1]

            # 绘制雷达图
            ax.plot(angles, model_avg_metrics, 'o-', linewidth=2, label=model)
            ax.fill(angles, model_avg_metrics, alpha=0.1)

        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(key_metric_names)
        ax.set_title('关键指标雷达图对比（前3个模型）')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

    # 12.4 精确率vs召回率散点图
    if detailed_results:
        # 提取所有精确率和召回率（宏平均）
        precisions = []
        recalls = []
        f1_scores = []
        model_names_list = []

        for result in detailed_results:
            precisions.append(result['precision_macro'])
            recalls.append(result['recall_macro'])
            f1_scores.append(result['f1_macro'])
            model_names_list.append(result['model'])

        # 创建散点图，点的大小表示F1分数
        scatter = axes[1, 1].scatter(precisions, recalls, s=np.array(f1_scores) * 200,
                                     alpha=0.6, c=f1_scores, cmap='viridis')
        axes[1, 1].set_title('精确率(宏) vs 召回率(宏) (点大小表示F1分数)')
        axes[1, 1].set_xlabel('精确率(宏平均)')
        axes[1, 1].set_ylabel('召回率(宏平均)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim([0, 1])
        axes[1, 1].set_ylim([0, 1])

        # 添加对角线
        axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='精确率=召回率')

        # 添加颜色条
        plt.colorbar(scatter, ax=axes[1, 1], label='F1分数(宏平均)')
        axes[1, 1].legend()

        # 标记最佳组合
        if f1_scores:
            best_idx = np.argmax(f1_scores)
            axes[1, 1].scatter(precisions[best_idx], recalls[best_idx],
                               color='red', s=200, edgecolors='black',
                               label=f'最佳F1: {model_names_list[best_idx]}')
            axes[1, 1].legend()

    plt.tight_layout()

    # 保存图表
    visualization_file = os.path.join(output_dir, 'all_metrics_visualization.png')
    plt.savefig(visualization_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"可视化图表已保存为: {visualization_file}")

except ImportError:
    print("\n注意: matplotlib 未安装，跳过可视化。如需可视化请运行: pip install matplotlib")
except Exception as e:
    print(f"\n可视化时出错: {e}")

print("\n评估完成!")
