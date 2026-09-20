"""
-*- coding: utf-8 -*-

@Author  : houcg
@Time    : 2026/1/23 16:52
"""

from config import *
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
import re
import logging

from utils import MistralAIEmbeddings
from utils.embedding_util import load_local_embeddings

logger = logging.getLogger(__name__)

csv = CSVLoader("./data/medical_knowledge_rag.csv", content_columns=["disease_name", "content"]).load()

print(csv[0].page_content[:], len(csv[0].page_content))
# 定义数据清洗模式
pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)

# 遍历所有文档
for doc in csv:
    # 对每个文档进行清洗
    doc.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), doc.page_content)

embedding_model = load_local_embeddings(
    model_path=EMBEDDING_MODEL_PATH,
    device='cpu')

embeddings_model = MistralAIEmbeddings()

# 构建向量数据框
db = FAISS.from_documents(csv, embeddings_model)

# # 保存向量数据库到本地
db.save_local(knowledge_path)
