import os.path

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

output_dir = "./output/model_eval"
df = pd.read_csv(os.path.join(output_dir, "model_results.csv"))

# 1. 首先删除 output 和 target 为 NaN 的行
df = df.dropna(subset=['output', 'target'])

print("原始数据示例:")
print(df[['model_name', 'output', 'target']].head())
print(f"\n总共有 {len(df)} 条数据")
print(f"模型名称: {df['model_name'].unique()}")

# 2. 创建全局类别映射
# 获取所有唯一的诊断名称（来自target和output）
all_categories = set()

# 从target和output中收集所有可能的诊断
for text in pd.concat([df['target'], df['output']]):
    all_categories.add(str(text).strip())

# 将类别转换为列表并排序
category_list = sorted(list(all_categories))
print(f"\n发现 {len(category_list)} 个唯一的诊断类别")

# 3. 创建全局标签编码器
label_encoder = LabelEncoder()
label_encoder.fit(category_list)


# 4. 定义一个函数来计算每个分组的指标
def calculate_metrics_for_model(model_df, label_encoder, model_name):
    """为单个模型计算评估指标"""
    if len(model_df) == 0:
        return None

    # 复制数据以避免修改原始数据
    model_df = model_df.copy()

    # 将文本转换为数字标签
    model_df['y_true'] = label_encoder.transform(model_df['target'])

    # 处理output中的未知类别
    def encode_output(x):
        x_str = str(x).strip()
        if x_str in label_encoder.classes_:
            return label_encoder.transform([x_str])[0]
        else:
            # 如果不在已知类别中，返回-1
            return -1

    model_df['y_pred'] = model_df['output'].apply(encode_output)

    # 过滤掉预测为未知类别的样本
    valid_mask = model_df['y_pred'] != -1
    if not valid_mask.any():
        print(f"  模型 {model_name}: 没有有效的预测样本")
        return None

    model_df_valid = model_df[valid_mask]
    y_true = model_df_valid['y_true'].values
    y_pred = model_df_valid['y_pred'].values

    # 计算样本数量
    n_samples = len(y_true)
    n_unknown = (~valid_mask).sum()

    # 计算评估指标
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

        return {
            'model_name': model_name,
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
        print(f"  模型 {model_name}: 计算指标时出错 - {e}")
        return None


# 5. 按 model_name 分组并计算指标
results = []

# 检查是否有 model_name 列
if 'model_name' not in df.columns:
    print("错误: DataFrame 中没有 'model_name' 列")
    # 如果没有 model_name 列，就计算整个数据集的指标
    result = calculate_metrics_for_model(df, label_encoder, "all_models")
    if result:
        results.append(result)
else:
    # 按 model_name 分组
    grouped = df.groupby('model_name')
    print(f"\n发现 {len(grouped)} 个不同的模型:")

    for model_name, group in grouped:
        print(f"  处理模型: {model_name} (样本数: {len(group)})")
        result = calculate_metrics_for_model(group, label_encoder, model_name)
        if result:
            results.append(result)
        else:
            print(f"  模型 {model_name}: 跳过，无有效数据")

# 6. 将结果转换为 DataFrame
if results:
    results_df = pd.DataFrame(results)

    # 重新排列列的顺序
    column_order = ['model_name', 'n_samples', 'n_unknown', 'accuracy',
                    'precision_macro', 'recall_macro', 'f1_macro',
                    'precision_micro', 'recall_micro', 'f1_micro',
                    'precision_weighted', 'recall_weighted', 'f1_weighted']
    results_df = results_df[column_order]

    print("\n" + "=" * 80)
    print("各模型评估结果")
    print("=" * 80)

    # 打印详细的评估结果
    for _, row in results_df.iterrows():
        print(f"\n模型: {row['model_name']}")
        print(f"样本数: {row['n_samples']} (未知预测: {row['n_unknown']})")
        print(f"准确率: {row['accuracy']:.4f}")
        print(
            f"宏平均 - 精确率: {row['precision_macro']:.4f}, 召回率: {row['recall_macro']:.4f}, F1分数: {row['f1_macro']:.4f}")
        print(
            f"微平均 - 精确率: {row['precision_micro']:.4f}, 召回率: {row['recall_micro']:.4f}, F1分数: {row['f1_micro']:.4f}")
        print(
            f"加权平均 - 精确率: {row['precision_weighted']:.4f}, 召回率: {row['recall_weighted']:.4f}, F1分数: {row['f1_weighted']:.4f}")

    # 7. 保存结果到CSV文件
    output_file = os.path.join(output_dir, "model_evaluation_results.csv")
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n评估结果已保存到: {output_file}")

    # 8. 打印汇总统计
    print("\n" + "=" * 80)
    print("汇总统计")
    print("=" * 80)

    # 按准确率排序
    sorted_by_acc = results_df.sort_values('accuracy', ascending=False)
    print("\n按准确率排序:")
    for i, (_, row) in enumerate(sorted_by_acc.iterrows()):
        print(f"  {i + 1}. {row['model_name']}: {row['accuracy']:.4f}")

    # 按F1分数（宏平均）排序
    sorted_by_f1 = results_df.sort_values('f1_macro', ascending=False)
    print("\n按F1分数（宏平均）排序:")
    for i, (_, row) in enumerate(sorted_by_f1.iterrows()):
        print(f"  {i + 1}. {row['model_name']}: {row['f1_macro']:.4f}")

    # 计算平均指标
    avg_accuracy = results_df['accuracy'].mean()
    avg_f1_macro = results_df['f1_macro'].mean()
    avg_f1_weighted = results_df['f1_weighted'].mean()

    print(f"\n所有模型平均指标:")
    print(f"  平均准确率: {avg_accuracy:.4f}")
    print(f"  平均F1分数（宏平均）: {avg_f1_macro:.4f}")
    print(f"  平均F1分数（加权平均）: {avg_f1_weighted:.4f}")

    # 9. 可视化比较
    try:
        import matplotlib.pyplot as plt

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 准确率比较
        models = results_df['model_name']
        x = range(len(models))

        axes[0, 0].bar(x, results_df['accuracy'])
        axes[0, 0].set_title('各模型准确率比较')
        axes[0, 0].set_xlabel('模型')
        axes[0, 0].set_ylabel('准确率')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models, rotation=45, ha='right')

        # F1分数比较（宏平均）
        axes[0, 1].bar(x, results_df['f1_macro'])
        axes[0, 1].set_title('各模型F1分数（宏平均）比较')
        axes[0, 1].set_xlabel('模型')
        axes[0, 1].set_ylabel('F1分数（宏平均）')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models, rotation=45, ha='right')

        # 精确率和召回率比较（宏平均）
        width = 0.35
        axes[1, 0].bar([i - width / 2 for i in x], results_df['precision_macro'], width, label='精确率')
        axes[1, 0].bar([i + width / 2 for i in x], results_df['recall_macro'], width, label='召回率')
        axes[1, 0].set_title('各模型精确率和召回率（宏平均）比较')
        axes[1, 0].set_xlabel('模型')
        axes[1, 0].set_ylabel('分数')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[1, 0].legend()

        # 样本数量
        axes[1, 1].bar(x, results_df['n_samples'])
        axes[1, 1].set_title('各模型样本数量')
        axes[1, 1].set_xlabel('模型')
        axes[1, 1].set_ylabel('样本数')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(models, rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\n可视化图表已保存为: {os.path.join(output_dir, 'model_comparison.png')}")

    except ImportError:
        print("\n注意: matplotlib 未安装，跳过可视化。如需可视化请运行: pip install matplotlib")
    except Exception as e:
        print(f"\n可视化时出错: {e}")
else:
    print("\n没有可用的评估结果")
