"""
-*- coding: utf-8 -*-

@Author  : houcg
@Time    : 2026/1/8 09:57
"""
import os

from openai import OpenAI

from utils.embedding_util import load_local_embeddings

ROOT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def _load_env_file(path):
    """读取 .env 文件并写入环境变量，已存在的变量不覆盖"""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file(os.path.join(ROOT_PATH, ".env"))

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.vectorengine.ai/v1"),
)

EMBEDDING_MODEL_PATH = os.path.join(ROOT_PATH, "models/bge-m3")
knowledge_path = os.path.join(ROOT_PATH, "knowledge_base")

if not os.path.exists(knowledge_path):
    os.makedirs(knowledge_path)

embedding_model = load_local_embeddings(
    model_path=EMBEDDING_MODEL_PATH,
    device='cpu')

model_config = {
    "gemini-3-pro-preview": {
        "role": "Imaging Department + Laboratory Department",
        "main": True,
        "prompt": """
### Task
You are a doctor specializing in radiology and laboratory medicine. Now you need to attend a consultation with two other doctors. You will be the chief doctor here. You need to reach the patient's diagnosis result through multiple rounds of conversations and discussions. Whether the result is correct or not is up to you to judge. When you believe the diagnosis result is correct, you need to output the summary and the diagnosis result.
### Output structure
    '''
        {
            "is_final": 0,
            "analysis_opinion": "xxx",
            "diagnosis_result": ""
        }
    '''

### Field definition
    is_final: Do you think the correct diagnosis has been discussed at present? The value is 0 or 1. 0 indicates no correct diagnosis has been discussed, and 1 indicates a correct diagnosis has been discussed
    summary: Your analysis of the patient's relevant information
    diagnosis_result: Diagnosis result, that is, the name of the disease
"""
    },
    "gpt-5.2": {
        "role": "",
        "main": False,
        "prompt": """
### Task
    You are a doctor. You are currently participating in a consultation and need to make an analysis and diagnosis based on the patient's condition description and relevant pictures. Of course, the leading physician and another physician will also have their own opinions. Please carefully analyze the condition
"""
    },
    "claude-opus-4-5-20251101": {
        "role": "Pathology Department + Oncology Department",
        "main": False,
        "prompt": """
### Task
    You are a doctor specializing in Pathology Department and Oncology Department. You are currently participating in a consultation and need to make an analysis and diagnosis based on the patient's condition description and relevant pictures. Of course, the leading physician and another physician will also have their own opinions. Please carefully analyze the condition
"""
    }
}
