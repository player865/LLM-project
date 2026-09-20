# 数据来源

## 网盘数据

通过网盘分享的知识：**LLM**

- 链接：https://pan.baidu.com/s/5zgwl8X8MYxpfu6gj5T2InA

## 病例图文数据

### NEJM Image Challenge

- 标注文件：`data/nejm_image_challenge.xlsx`
- 图片目录：`data/images/nejm_images/`（523 张）
- 来源：NEJM Image Challenge

### The Lancet 临床病例图片

- 标注文件：`data/lancet_data(total).xlsx`
- 图片目录：`data/images/lancet_images/`（511 张）
- 来源：The Lancet

### 训练 / 测试集

- 文件：`data/train_with_options.csv`、`data/test_with_options.csv`
- 由 `data_process.py` 基于上述 NEJM、Lancet 两份数据合并后按 7:3 划分生成

## 医学知识库（RAG）

- 原始文件：`data/medical_knowledge_rag.csv`
  - 共 24,353 条，字段：`disease_name`、`disease_id`、`source`、`content`
  - `source` 分布：HPO（13,009）、DiseaseOntology（10,328）、MedlinePlus（1,016）
- 向量索引：`knowledge_base/`，由 `vector_create.py` 构建
