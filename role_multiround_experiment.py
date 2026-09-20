"""
-*- coding: utf-8 -*-

@Author  : houcg
@Time    : 2026/1/8 11:11
Multi-model collaborative diagnosis with role permutation experiment
每个模型轮流扮演每个角色，形成完整的排列组合实验
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
import re
from itertools import permutations

should_exit = False
exit_event = threading.Event()

train_data = pd.read_csv("./data/train_with_options.csv")
train_data = train_data.iloc[:200, :]
train_data.reset_index(drop=True, inplace=True)

# 角色定义
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

# 角色配置：包含主管理模型（""）和三个科室角色
roles = [
    "",  # 主管理模型
    "Imaging Department + Laboratory Department",  # 影像科 + 检验科
    "Pathology Department + Oncology Department"   # 病理科 + 肿瘤科
]

ROLE_NAMES = {
    "": "主管理模型",
    "Imaging Department + Laboratory Department": "影像科 + 检验科",
    "Pathology Department + Oncology Department": "病理科 + 肿瘤科"
}

# 讨论流程 Prompt
DISCUSSION_INIT_PROMPT = """
### Case Information
**Clinical Description**: {clinical_description}

**Imaging Data**: {imaging_description}

### Multiple Choice Options
{options}

### Task
You are part of a multi-disciplinary team. Please introduce yourself and share your initial assessment based on your expertise.
"""

DISCUSSION_ROUND_PROMPT = """
### Previous Discussion
{discussion_history}

### Task
Based on the discussion above, please provide your updated assessment. Consider other specialists' opinions and explain how they influence your diagnosis.

If you agree with a specific diagnosis from the options, clearly state your agreement and reasoning.
"""

FINAL_CONSENSUS_PROMPT = """
### Complete Discussion History
{discussion_history}

### Task
Based on the entire discussion, please provide your final diagnostic conclusion. Select one option from the multiple choice options and explain your reasoning.

If there is disagreement among specialists, acknowledge it but still provide your best assessment.
"""

models = [
    {
        "name": "claude",
        "model": "claude-opus-4-5-20251101"
    },
    {
        "name": "gemini",
        "model": "gemini-3-pro-preview"
    },
    {
        "name": "deepseek",
        "model": "deepseek-v3-1-think-250821",
        "assist_model": "claude-opus-4-5-20251101"
    }
]

roles = ["radiologist", "pathologist", "oncologist"]

external_roles = [""]  # 不使用外部角色提示，直接使用模型分配的角色


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


def parse_model_output(output):
    try:
        diagnosis_match = re.search(r'(?:diagnosis|selection|answer|conclusion)[:\s]+(.+?)(?:\n|$)', output, re.IGNORECASE)
        diagnosis = diagnosis_match.group(1).strip() if diagnosis_match else output.strip()

        questions_match = re.findall(r'(?:question|\d+)\.?\s*(.+?)(?:\n|$)', output, re.IGNORECASE)
        questions = [q.strip() for q in questions_match[:3]]

        return {
            "diagnosis": diagnosis,
            "questions": questions
        }
    except Exception as e:
        logger.error(f"Failed to parse output: {e}")
        return None


def run_collaborative_diagnosis(role_assignment, content, image_path, target, openai_client, encode_image_func, options=None, external_role=""):
    """
    role_assignment: dict, key=model_name, value=role (e.g., {"deepseek": "", "claude": "Imaging Department + Laboratory Department", "gemini": "Pathology Department + Oncology Department"})
    """
    try:
        options_text = str(options).strip() if options else ""

        image_base64 = encode_image_func(image_path)

        # 使用外部角色 prompt（如果有）
        base_role_prompt = DEFAULT_PROMPT if external_role == "" else ROLE_PROMPT.replace("role_template", external_role)

        discussion_history = []
        model_opinions = []
        max_rounds = 3

        for round_num in range(1, max_rounds + 1):
            if should_exit or exit_event.is_set():
                break

            round_outputs = []

            for model_info in models:
                model_name = model_info["name"]
                model_id = model_info["model"]
                assigned_role = role_assignment[model_name]
                assist_model = model_info.get("assist_model")

                # 构建模型的系统 prompt
                if assigned_role == "":
                    # 主管理模型，使用 DEFAULT_PROMPT
                    system_prompt = base_role_prompt
                else:
                    # 科室角色，使用 ROLE_PROMPT
                    system_prompt = base_role_prompt + "\n\n" + ROLE_PROMPT.replace("role_template", assigned_role)

                image_description = ""
                if assist_model:
                    logger.info(f"Using {assist_model} to describe image for {model_name}")
                    image_description = describe_image_with_model(image_base64, assist_model, openai_client)

                if image_description:
                    imaging_data_text = f"**Image Description**:\n{image_description}"
                else:
                    imaging_data_text = "(See attached image)"

                init_prompt = DISCUSSION_INIT_PROMPT.format(
                    clinical_description=content,
                    imaging_description=imaging_data_text,
                    options=options_text
                )

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": init_prompt}
                ]

                if not image_description:
                    messages[1]["content"] = [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpg;base64,{image_base64}"
                            },
                        },
                        {"type": "text", "text": init_prompt},
                    ]

                if round_num > 1:
                    discussion_text = "\n\n".join([
                        f"### {ROLE_NAMES[op['role']]} - {op['model']} (Round {op['round']})\n{op['raw_output']}"
                        for op in discussion_history
                    ])

                    next_prompt = DISCUSSION_ROUND_PROMPT.format(
                        discussion_history=discussion_text
                    )

                    messages.append({"role": "assistant", "content": "Previous response incorporated"})
                    messages.append({"role": "user", "content": next_prompt})

                output = model_process(openai_client, messages, model_id)
                if not output:
                    continue

                parsed = parse_model_output(output)

                opinion = {
                    "round": round_num,
                    "model": model_name,
                    "model_id": model_id,
                    "role": assigned_role,
                    "role_name": ROLE_NAMES[assigned_role],
                    "raw_output": output,
                    "parsed": parsed
                }
                round_outputs.append(opinion)

                if parsed:
                    diagnosis = parsed.get("diagnosis", "")
                    model_opinions.append(diagnosis)

            discussion_history.extend(round_outputs)

        # Final consensus round
        final_consensus_prompt = FINAL_CONSENSUS_PROMPT.format(
            discussion_history="\n\n".join([
                f"### {op['role_name']} - {op['model']} (Round {op['round']})\n{op['raw_output']}"
                for op in discussion_history
            ])
        )

        final_opinions = []
        for model_info in models:
            model_name = model_info["name"]
            model_id = model_info["model"]
            assigned_role = role_assignment[model_name]

            if assigned_role == "":
                system_prompt = base_role_prompt
            else:
                system_prompt = base_role_prompt + "\n\n" + ROLE_PROMPT.replace("role_template", assigned_role)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_consensus_prompt}
            ]

            final_output = model_process(openai_client, messages, model_id)
            final_parsed = parse_model_output(final_output) if final_output else None
            final_diagnosis = final_parsed.get("diagnosis", "") if final_parsed else ""

            if not final_diagnosis and model_opinions:
                final_diagnosis = model_opinions[-1]

            final_opinions.append({
                "model": model_name,
                "model_id": model_id,
                "role": assigned_role,
                "role_name": ROLE_NAMES[assigned_role],
                "final_diagnosis": final_diagnosis,
                "final_output": final_output
            })

        return {
            "final_opinions": final_opinions,
            "discussion_history": discussion_history,
            "num_rounds": len(discussion_history)
        }

    except Exception as e:
        logger.error(f"Collaborative diagnosis failed: {str(e)}")
        return {"error": str(e)}


def create_task_id(role_assignment, data_index, content, external_role):
    role_str = "_".join([f"{k}:{v}" for k, v in sorted(role_assignment.items())])
    task_str = f"{role_str}_{data_index}_{content[:50]}_{external_role[:20]}"
    return hashlib.md5(task_str.encode()).hexdigest()[:16]


def generate_role_assignment_id(role_assignment):
    """
    Generate a readable ID for the role assignment
    e.g., "claude=影像科|gemini=病理科|deepseek=肿瘤科"
    """
    return "|".join([f"{m}={ROLE_NAMES[role_assignment[m]]}" for m in ["claude", "gemini", "deepseek"]])


def evaluate_model_wrapper(role_assignment, content, image_path, target, task_id, openai_client,
                          encode_image_func, options=None, external_role=""):
    try:
        result = run_collaborative_diagnosis(
            role_assignment=role_assignment,
            content=content,
            image_path=image_path,
            target=target,
            openai_client=openai_client,
            encode_image_func=encode_image_func,
            options=options,
            external_role=external_role
        )

        if "error" in result:
            error_records = []
            role_assignment_id = generate_role_assignment_id(role_assignment)
            for model_info in models:
                error_records.append({
                    "task_id": task_id,
                    "external_role": external_role,
                    "role_assignment_id": role_assignment_id,
                    "model_name": model_info["name"],
                    "model_id": model_info["model"],
                    "assigned_role": role_assignment[model_info["name"]],
                    "role_name": ROLE_NAMES[role_assignment[model_info["name"]]],
                    "input": content,
                    "image_path": image_path,
                    "output": f"ERROR: {result['error']}",
                    "target": target,
                    "flag": False,
                    "status": "failed",
                    "conversation_rounds": 0,
                    "error": result.get("error"),
                    "timestamp": datetime.now().isoformat()
                })
            return error_records

        judge_model = "gpt-5-mini"
        role_assignment_id = generate_role_assignment_id(role_assignment)
        records = []

        for opinion in result["final_opinions"]:
            output_diagnosis = opinion["final_diagnosis"]
            flag = diagnostic_judge(openai_client, target, output_diagnosis, judge_model)

            records.append({
                "task_id": task_id,
                "external_role": external_role,
                "role_assignment_id": role_assignment_id,
                "model_name": opinion["model"],
                "model_id": opinion["model_id"],
                "assigned_role": opinion["role"],
                "role_name": opinion["role_name"],
                "input": content,
                "image_path": image_path,
                "output": output_diagnosis,
                "target": target,
                "flag": flag,
                "status": "success",
                "conversation_rounds": result["num_rounds"],
                "timestamp": datetime.now().isoformat()
            })

        return records

    except Exception as e:
        logger.error(f"Model processing failed: {str(e)}")
        error_records = []
        role_assignment_id = generate_role_assignment_id(role_assignment)
        for model_info in models:
            error_records.append({
                "task_id": task_id,
                "external_role": external_role,
                "role_assignment_id": role_assignment_id,
                "model_name": model_info["name"],
                "model_id": model_info["model"],
                "assigned_role": role_assignment[model_info["name"]],
                "role_name": ROLE_NAMES[role_assignment[model_info["name"]]],
                "input": content,
                "image_path": image_path,
                "output": f"ERROR: {str(e)}",
                "target": target,
                "flag": False,
                "status": "failed",
                "conversation_rounds": 0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
        return error_records


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


def generate_role_assignments():
    """
    Generate role assignments where each model takes each role once
    Since we have 3 models and 3 roles, we create 3 scenarios:
    
    Scenario 1: claude=主管理，gemini=影像 + 检验，deepseek=病理 + 肿瘤
    Scenario 2: claude=影像 + 检验，gemini=病理 + 肿瘤，deepseek=主管理
    Scenario 3: claude=病理 + 肿瘤，gemini=主管理，deepseek=影像 + 检验
    
    This ensures each model experiences each role exactly once.
    """
    role_assignments = []
    
    # Scenario 1
    role_assignments.append({
        "claude": "",
        "gemini": "Imaging Department + Laboratory Department",
        "deepseek": "Pathology Department + Oncology Department"
    })
    
    # Scenario 2
    role_assignments.append({
        "claude": "Imaging Department + Laboratory Department",
        "gemini": "Pathology Department + Oncology Department",
        "deepseek": ""
    })
    
    # Scenario 3
    role_assignments.append({
        "claude": "Pathology Department + Oncology Department",
        "gemini": "",
        "deepseek": "Imaging Department + Laboratory Department"
    })
    
    return role_assignments


def process_batch_parallel(openai_client, train_data, role_permutations, output_dir,
                          encode_image_func, max_workers=3, checkpoint_interval=5):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_file = os.path.join(output_dir, "role_permutation_results.csv")
    temp_result_file = os.path.join(output_dir, "role_permutation_results_temp.csv")

    completed_task_ids = get_completed_task_ids_from_file(temp_result_file)

    if len(completed_task_ids) == 0 and os.path.exists(result_file):
        logger.info(f"Temp file not found, checking result file: {result_file}")
        completed_task_ids = get_completed_task_ids_from_file(result_file)

    all_tasks = []

    for i in range(len(train_data)):
        content = train_data["input"][i]
        target = train_data["target"][i]
        image_path = train_data["image_path"][i]
        options = train_data["options"][i] if "options" in train_data.columns else None

        for role_assignment in role_permutations:
            # 不使用外部角色提示
            external_role = ""
            task_id = create_task_id(role_assignment, i, content, external_role)
            all_tasks.append({
                "task_id": task_id,
                "role_assignment": role_assignment,
                "external_role": external_role,
                "content": content,
                "image_path": image_path,
                "target": target,
                "options": options,
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
            "task_id", "external_role", "model_name", "model_id", "assigned_role", "role_name",
            "role_assignment_id", "input", "image_path", "output", "target", "flag", "status",
            "conversation_rounds", "timestamp", "error"
        ])

    pbar = tqdm(total=len(pending_tasks), desc="Processing progress")
    completed_count = len(completed_task_ids)
    lock = threading.Lock()

    evaluate_func = partial(
        evaluate_model_wrapper,
        openai_client=openai_client,
        encode_image_func=encode_image_func
    )

    def task_callback(future, task_detail):
        nonlocal completed_count, result_df, pbar, completed_task_ids

        try:
            results = future.result()
            task_id = task_detail["task_id"]

            with lock:
                completed_task_ids.add(task_id)
                completed_count += 1

                for result in results:
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
                task_detail["role_assignment"],
                task_detail["content"],
                task_detail["image_path"],
                task_detail["target"],
                task_detail["task_id"],
                options=task_detail.get("options"),
                external_role=task_detail.get("external_role")
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
    has_options = True

    dataset_name = "with_options" if has_options else "no_options"
    output_dir = f"./output/role_multiround_experiment_final"

    os.makedirs(output_dir, exist_ok=True)

    max_workers = 2
    checkpoint_interval = 3

    role_assignments = generate_role_assignments()

    print(f"Running role assignment experiment")
    print(f"Output directory: {output_dir}")
    print(f"Total samples: {len(train_data)}")
    print(f"Models: {[m['name'] for m in models]}")
    print(f"Roles: {roles}")
    print(f"Role names mapping: {ROLE_NAMES}")
    print(f"Number of role assignments: {len(role_assignments)}")
    print("-" * 60)
    print("Role assignment scenarios:")
    for i, assignment in enumerate(role_assignments, 1):
        print(f"  Scenario {i}: " + ", ".join([f"{m}={ROLE_NAMES[r]}" for m, r in assignment.items()]))
    print("-" * 60)
    print(f"Total experiments: {len(role_assignments)} role assignments × {len(train_data)} cases = {len(role_assignments) * len(train_data)} records")
    print("=" * 60)

    result_df = process_batch_parallel(
        openai_client=openai_client,
        train_data=train_data,
        role_permutations=role_assignments,
        output_dir=output_dir,
        encode_image_func=encode_image,
        max_workers=max_workers,
        checkpoint_interval=checkpoint_interval
    )

    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"Total records: {len(result_df)}")
    print(f"Success: {len(result_df[result_df['status'] == 'success'])}")
    print(f"Failed: {len(result_df[result_df['status'] == 'failed'])}")

    print(f"\nResults saved to: {os.path.join(output_dir, 'role_permutation_results.csv')}")

    if not result_df.empty:
        print("\n" + "=" * 60)
        print("Summary by Role Permutation")
        print("=" * 60)

        summary = result_df.groupby(['role_assignment_id', 'external_role']).agg({
            'flag': ['sum', 'count', 'mean'],
            'conversation_rounds': 'mean'
        }).round(4)
        summary.columns = ['correct', 'total', 'accuracy', 'avg_rounds']
        print(summary.to_string())

        print("\n" + "=" * 60)
        print("Best Performing Permutation (by accuracy)")
        print("=" * 60)
        best = summary.loc[summary['accuracy'].idxmax()]
        print(best.to_string())
