"""
-*- coding: utf-8 -*-

@Author  : houcg
@Time    : 2026/1/8 10:23
"""
from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel
from typing import List

from tqdm import tqdm

class MistralAIEmbeddings(BaseModel, Embeddings):

    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.

        Args:
            texts (str): 要生成 embedding 的文本.

        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表.
        """
        from config import embedding_model

        embeddings = embedding_model.embed_query(text)
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成输入文本列表的 embedding.

        Args:
            texts (List[str]): 要生成 embedding 的文本列表.

        Returns:
            List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        """
        result = []
        for i in tqdm(range(len(texts))):
            result.append(self.embed_query(texts[i]))
        return result
