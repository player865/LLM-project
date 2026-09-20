import re

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader , JSONLoader , CSVLoader
from langchain.text_splitter import CharacterTextSplitter


def load_local_embeddings(model_path: str = None, device='cpu'):
    '''
    从缓存中加载embeddings，可以避免多线程时竞争加载。
    '''

    query_instruction = "为这个句子生成表示以用于检索相关文章："
    embeddings = HuggingFaceBgeEmbeddings(model_name=model_path,
                                          model_kwargs={'device': device},
                                          query_instruction=query_instruction
                                          )
    return embeddings


def create_knowledge_base(file_path):
    if file_path.endswith('.txt'):
        loader = TextLoader(file_path , encoding="utf-8")
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.json'):
        loader = JSONLoader(file_path , '.')
    elif file_path.endswith('.csv'):
        loader = CSVLoader(file_path)
    docs = loader.load()
    pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
    # 遍历所有文档
    for doc in docs:
        # 对每个文档进行清洗
        doc.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), doc.page_content)
    text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=100,
                                          length_function=len,
                                          is_separator_regex=False, separator="\n")
    texts = text_splitter.split_documents(docs)
    return texts
