"""
-*- coding: utf-8 -*-

@Author  : houcg
@Time    : 2026/1/8 11:11
Role-based experiment with Claude image description + DeepSeek diagnosis
"""
import os

import pandas as pd
from config import openai_client
from loguru import logger
from utils.llm_util import encode_image, model_process, diagnostic_judge
import concurrent.futures
import threading
from tqdm import tqdm
import json
from datetime import datetime
from functools import partial
import signal
import hashlib

should_exit = False
exit_event = threading.Event()

train_data = pd.read_csv("./data/train_with_options.csv")

DEFAULT_PROMPT = """
### Task
You are a medical diagnostic model. Now you need to make a diagnosis based on the patient's condition description and related pictures. You only need to output the diagnostic result
### Example
    input:A woman aged 60 years presented, in 2023, with progressive deviation of her occlusion, although she experienced no pain. A panoramic x-ray revealed birateral osteolysis of the mandible. Contrast-enhanced CT demonstrated progressive mandibular bone resorption extending from the mandibular body to the condylar regions, prompting referral to the departments of rheumatology, haematology, and oral surgery for further evaluation. She had no other symptoms, and clinically there were no signs of arthritis (ie, no joint swelling or tenderness). However, hand radiographs obtained as part of systemic screening prompted by the mandibular findings subsequently revealed an unusual pattern of osteolysis. Laboratory tests showed a normal C-reactive protein concentration (0·10 mg/dL) and matrix metalloproteinase-3 (20·8 ng/mL), although tartrate-resistant acid phosphatase type 5b was markedly elevated (851 U/L), suggesting substantial osteoclastic activity. Antinuclear antibody was positive at 1:320 (speckled and cytoplasmic pattern) and anti-Sjögren's syndrome-related antigen A antibody was also positive, raising the suspicion of underlying Sjögren's disease. Immunofixation electrophoresis revealed no monoclonal protein, ruling out multiple myeloma. Other skeletal disorders (eg, enchondromatosis), metabolic disorders (eg, hyperparathyroidism), and rheumatic diseases (eg, sarcoidosis, psoriatic arthritis, and rheumatoid arthritis) were also excluded. Biopsy of the mandible showed massive osteolysis and abnormal lymphatic proliferation. In combination with the progressive osteolysis demonstrated radiographically, the patient was diagnosed as Gorham–Stout disease.
    output:Segmental osteolytic lesions progressing in a centrifugal pattern without joint specificity

    input:A 12-year-old boy with a history of obesity presented with a 2-week history of limping and dull pain in the right hip. The symptoms had begun after he had slipped and fallen at school. On physical exam, active and passive ranges of motion of the right hip were limited by pain. Radiographs of the pelvis were performed (frog-leg lateral view is shown). What is the most likely diagnosis?
    output:Slipped capital femoral epiphysis
### Attention
    1、You only need to output the diagnosis result. No other descriptions are allowed
"""

ROLE_PROMPT = """
### Role
    role_template
### Task
You are a medical diagnostic model. Now you need to make a diagnosis based on the patient's condition description and related pictures. You only need to output the diagnostic result
### Example
    input:A woman aged 60 years presented, in 2023, with progressive deviation of her occlusion, although she experienced no pain. A panoramic x-ray revealed birateral osteolysis of the mandible. Contrast-enhanced CT demonstrated progressive mandibular bone resorption extending from the mandibular body to the condylar regions, prompting referral to the departments of rheumatology, haematology, and oral surgery for further evaluation. She had no other symptoms, and clinically there were no signs of arthritis (ie, no joint swelling or tenderness). However, hand radiographs obtained as part of systemic screening prompted by the mandibular findings subsequently revealed an unusual pattern of osteolysis. Laboratory tests showed a normal C-reactive protein concentration (0·10 mg/dL) and matrix metalloproteinase-3 (20·8 ng/mL), although tartrate-resistant acid phosphatase type 5b was markedly elevated (851 U/L), suggesting substantial osteoclastic activity. Antinuclear antibody was positive at 1:320 (speckled and cytoplasmic pattern) and anti-Sjögren's syndrome-related antigen A antibody was also positive, raising the suspicion of underlying Sjögren's disease. Immunofixation electrophoresis revealed no monoclonal protein, ruling out multiple myeloma. Other skeletal disorders (eg, enchondromatosis), metabolic disorders (eg, hyperparathyroidism), and rheumatic diseases (eg, sarcoidosis, psoriatic arthritis, and rheumatoid arthritis) were also excluded. Biopsy of the mandible showed massive osteolysis and abnormal lymphatic proliferation. In combination with the progressive osteolysis demonstrated radiographically, the patient was diagnosed as Gorham–Stout disease.
    output:Segmental osteolytic lesions progressing in a centrifugal pattern without joint specificity

    input:A 12-year-old boy with a history of obesity presented with a 2-week history of limping and dull pain in the right hip. The symptoms had begun after he had slipped and fallen at school. On physical exam, active and passive ranges of motion of the right hip were limited by pain. Radiographs of the pelvis were performed (frog-leg lateral view is shown). What is the most likely diagnosis?
    output:Slipped capital femoral epiphysis
### Attention
    1、You only need to output the diagnosis result. No other descriptions are allowed
"""

assist_model = "claude-opus-4-5-20251101"
diagnostic_model = "deepseek-v3-1-think-250821"
judge_model = "gpt-5-mini"

roles = ["", "Imaging Department + Laboratory Department", "Pathology Department + Oncology Department"]


def signal_handler(signum, frame):
    global should_exit
    logger.info("Received interrupt signal, saving progress...")
    should_exit = True
    exit_event.set()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def describe_image_with_model(image_base64, assist_model, openai_client):
    image_description_prompt = """Please describe this medical image in detail, including:
1. What type of imaging modality is this?
2. What anatomical structures are visible?
3. What abnormalities or findings can be observed?
4. Any notable features that would be relevant for diagnosis?

Please provide a detailed and objective description."""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpg;base64,{image_base64}"
                    },
                },
                {"type": "text", "text": image_description_prompt},
            ],
        },
    ]

    description = model_process(openai_client, messages, assist_model)
    if not description:
        logger.warning(f"Failed to get image description from {assist_model}")
        return ""
    return description


def create_task_id(role, model_name, data_index, content):
    task_str = f"{role}_{model_name}_{data_index}_{content[:100]}"
    return hashlib.md5(task_str.encode()).hexdigest()[:12]


def evaluate_model_wrapper(role, model_name, content, image_path, target, task_id, openai_client,
                          encode_image_func, assist_model, diagnostic_model, judge_model):
    try:
        image_base64 = encode_image_func(image_path)

        logger.info(f"Using {assist_model} to describe image")
        image_description = describe_image_with_model(image_base64, assist_model, openai_client)

        PROMPT = DEFAULT_PROMPT if role == "" else ROLE_PROMPT.replace("role_template", role)

        if image_description:
            imaging_data_text = f"**Image Description**:\n{image_description}"
        else:
            imaging_data_text = "(See attached image)"

        conversation_history = f"""
### Case Information
**Clinical Description**: {content}

**Imaging Data**: {imaging_data_text}
"""

        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": conversation_history}
        ]

        output = model_process(openai_client, messages, diagnostic_model)

        flag = diagnostic_judge(openai_client, target, output, judge_model)

        if flag:
            return {
                "task_id": task_id,
                "role": role,
                "assist_model": assist_model,
                "diagnostic_model": diagnostic_model,
                "input": content,
                "image_path": image_path,
                "output": target,
                "target": target,
                "flag": flag,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "task_id": task_id,
                "role": role,
                "assist_model": assist_model,
                "diagnostic_model": diagnostic_model,
                "input": content,
                "image_path": image_path,
                "output": output,
                "target": target,
                "flag": flag,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Model {model_name} processing failed: {str(e)}")
        return {
            "task_id": task_id,
            "role": role,
            "assist_model": assist_model,
            "diagnostic_model": diagnostic_model,
            "input": content,
            "image_path": image_path,
            "output": f"ERROR: {str(e)}",
            "target": target,
            "flag": False,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def get_completed_task_ids_from_file(result_file):
    completed_ids = set()

    if os.path.exists(result_file):
        try:
            existing_df = pd.read_csv(result_file)
            if not existing_df.empty and 'task_id' in existing_df.columns:
                completed_ids = set(existing_df['task_id'].dropna().astype(str).tolist())
                logger.info(f"Loaded {len(completed_ids)} completed tasks from result file")
        except Exception as e:
            logger.warning(f"Failed to read result file: {str(e)}")

    return completed_ids


def process_batch_parallel(openai_client, train_data, roles, output_dir,
                          encode_image_func, assist_model, diagnostic_model, judge_model,
                          max_workers=3, checkpoint_interval=5):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_file = os.path.join(output_dir, "model_results.csv")
    temp_result_file = os.path.join(output_dir, "model_results_temp.csv")

    completed_task_ids = get_completed_task_ids_from_file(temp_result_file)

    if len(completed_task_ids) == 0 and os.path.exists(result_file):
        logger.info(f"Temp file not found, checking result file: {result_file}")
        completed_task_ids = get_completed_task_ids_from_file(result_file)

    all_tasks = []

    for i in range(len(train_data)):
        content = train_data["input"][i]
        target = train_data["target"][i]
        image_path = train_data["image_path"][i]

        for role in roles:
            task_id = create_task_id(role, diagnostic_model, i, content)
            all_tasks.append({
                "task_id": task_id,
                "role": role,
                "content": content,
                "image_path": image_path,
                "target": target,
                "data_index": i
            })

    pending_tasks = [task for task in all_tasks if task["task_id"] not in completed_task_ids]

    logger.info(f"Total tasks: {len(all_tasks)}, Completed: {len(completed_task_ids)}, Pending: {len(pending_tasks)}")

    if len(pending_tasks) == 0:
        logger.info("All tasks completed, skipping")
        if os.path.exists(result_file):
            return pd.read_csv(result_file)
        return pd.DataFrame()

    if os.path.exists(temp_result_file) and len(completed_task_ids) > 0:
        result_df = pd.read_csv(temp_result_file)
    elif os.path.exists(result_file) and len(completed_task_ids) > 0:
        logger.info(f"Loading result from result file: {result_file}")
        result_df = pd.read_csv(result_file)
    else:
        result_df = pd.DataFrame(columns=[
            "task_id", "role", "assist_model", "diagnostic_model", "input", "image_path", "output",
            "target", "flag", "status", "timestamp", "error"
        ])

    pbar = tqdm(total=len(pending_tasks), desc="Processing progress")
    completed_count = len(completed_task_ids)
    lock = threading.Lock()

    evaluate_func = partial(
        evaluate_model_wrapper,
        openai_client=openai_client,
        encode_image_func=encode_image_func,
        assist_model=assist_model,
        diagnostic_model=diagnostic_model,
        judge_model=judge_model
    )

    def task_callback(future, task_detail):
        nonlocal completed_count, result_df, pbar, completed_task_ids

        try:
            result = future.result()
            task_id = task_detail["task_id"]

            with lock:
                completed_task_ids.add(task_id)
                completed_count += 1

                new_row = pd.DataFrame([result])
                result_df = pd.concat([result_df, new_row], ignore_index=True)

                pbar.update(1)
                success_count = len(result_df[result_df['status'] == 'success'])
                failed_count = len(result_df[result_df['status'] == 'failed'])
                pbar.set_postfix({
                    "Completed": completed_count,
                    "Success Rate": f"{success_count / completed_count * 100:.1f}%" if completed_count > 0 else "0%"
                })

                if completed_count % checkpoint_interval == 0 or exit_event.is_set():
                    try:
                        result_df.to_csv(temp_result_file, index=False, encoding='utf-8')
                        logger.debug(f"Temp results saved: {temp_result_file}")
                    except Exception as e:
                        logger.error(f"Failed to save temp results: {str(e)}")

                if should_exit or exit_event.is_set():
                    logger.info("Exiting gracefully...")
                    result_df.to_csv(result_file, index=False, encoding='utf-8')
                    logger.info(f"Results saved to: {result_file}")
                    pbar.close()
                    return

        except Exception as e:
            logger.error(f"Task callback processing failed: {str(e)}")
            with lock:
                completed_count += 1
                pbar.update(1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for task_detail in pending_tasks:
            if should_exit or exit_event.is_set():
                logger.info("Exit signal detected, stopping task submission")
                break

            future = executor.submit(
                evaluate_func,
                task_detail["role"],
                task_detail.get("model_name", diagnostic_model),
                task_detail["content"],
                task_detail["image_path"],
                task_detail["target"],
                task_detail["task_id"]
            )

            future.add_done_callback(
                lambda f, td=task_detail: task_callback(f, td)
            )

            futures.append(future)

            if len(futures) >= max_workers * 2:
                concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                futures = [f for f in futures if not f.done()]

        if not should_exit and not exit_event.is_set():
            concurrent.futures.wait(futures)

    pbar.close()

    try:
        if not result_df.empty:
            result_df.to_csv(result_file, index=False, encoding='utf-8')
            logger.info(f"Final results saved to: {result_file}")

            if os.path.exists(temp_result_file):
                os.remove(temp_result_file)

            success_count = len(result_df[result_df['status'] == 'success'])
            failed_count = len(result_df[result_df['status'] == 'failed'])
            logger.info(f"Processing complete! Success: {success_count}, Failed: {failed_count}")

    except Exception as e:
        logger.error(f"Failed to save final results: {str(e)}")
        if os.path.exists(temp_result_file):
            os.rename(temp_result_file, result_file)
            logger.info(f"Results recovered from temp file: {result_file}")

    return result_df


if __name__ == '__main__':
    has_options = False

    dataset_name = "with_options" if has_options else "no_options"
    output_dir = f"./output/role_eval"

    os.makedirs(output_dir, exist_ok=True)

    max_workers = 2
    checkpoint_interval = 3

    print(f"Running role experiment - Claude image description + DeepSeek diagnosis")
    print(f"Output directory: {output_dir}")
    print(f"Total samples: {len(train_data)}")
    print(f"Assist model: {assist_model}")
    print(f"Diagnostic model: {diagnostic_model}")
    print(f"Roles: {roles}")
    print("-" * 60)

    result_df = process_batch_parallel(
        openai_client=openai_client,
        train_data=train_data,
        roles=roles,
        output_dir=output_dir,
        encode_image_func=encode_image,
        assist_model=assist_model,
        diagnostic_model=diagnostic_model,
        judge_model=judge_model,
        max_workers=max_workers,
        checkpoint_interval=checkpoint_interval
    )

    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"Total tasks: {len(result_df)}")
    print(f"Success: {len(result_df[result_df['status'] == 'success'])}")
    print(f"Failed: {len(result_df[result_df['status'] == 'failed'])}")

    print(f"\nResults saved to: {os.path.join(output_dir, 'role_deepseek_results.csv')}")
