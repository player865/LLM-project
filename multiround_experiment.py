"""
-*- coding: utf-8 -*-

@Author  : houcg
@Time    : 2026/1/26 10:00
"""

"""
单模型多轮对话试验
"""
from utils import MistralAIEmbeddings

embeddings_model = MistralAIEmbeddings()

import os
import json
import re
import pandas as pd
import hashlib
import concurrent.futures
import threading
from datetime import datetime
from functools import partial
from tqdm import tqdm
import signal

from config import openai_client, knowledge_path
from utils.llm_util import encode_image, model_process, diagnostic_judge
from langchain_community.vectorstores import FAISS
from loguru import logger

should_exit = False
exit_event = threading.Event()

SELF_DIALOGUE_PROMPT = """
### Role Setting
You are a rigorous medical expert conducting self-diagnosis and reflection on a clinical case.

### Task Requirements
You need to complete the diagnostic task through multiple rounds of self-dialogue. Each round must complete the following steps:

## Round 1: Initial Diagnosis
Based on the case description and imaging data, provide your preliminary diagnosis.

## Round 2: Critical Review
Switch to a critical perspective and review your initial diagnosis:
1. What is the most important supporting evidence?
2. What are the most likely differential diagnoses?
3. What additional information would be most valuable for confirmation?

## Round 3: Knowledge Retrieval and Verification
Based on the questions from the previous round, retrieve relevant medical knowledge, then provide a more accurate diagnosis.

## Round 4: Final Confirmation
Synthesize all information to provide the final diagnosis with diagnostic basis.

### Output Format
After each round, please output in the following format:

```json
{
    "round": 1,
    "type": "initial_diagnosis/preliminary_criticism/knowledge_retrieval/final_confirmation",
    "reasoning": "Your analytical reasoning process",
    "diagnosis": "Diagnostic result",
    "confidence": 0.8,
    "self_questions": ["List of questions for self-inquiry"],
    "needs_retrieval": true/false
}
```

### Important Notes
1. Honestly assess your diagnostic confidence in each round
2. Proactively identify your knowledge gaps and reasoning vulnerabilities
3. When identifying need for more information, clearly state retrieval requirements
4. Final diagnosis must be based on sufficient reasoning and evidence

Please start Round 1.
"""

SELF_CRITICAL_PROMPT = """
### Task
Please critically review your initial diagnosis.

### Analysis Dimensions
1. **Supporting Evidence Analysis**: List all evidence supporting this diagnosis, including symptoms, signs, imaging findings, etc.
2. **Differential Diagnoses**: List differential diagnoses that need to be ruled out or considered, with reasons for each
3. **Information Gaps**: Clearly identify the most important information missing for confirmation
4. **Retrieval Needs**: Based on the above analysis, propose medical knowledge topics that need to be retrieved

### Output Requirements
Please output in JSON format:
```json
{
    "supporting_evidence": ["List of evidence"],
    "differential_diagnoses": [{"diagnosis": "Diagnosis name", "likelihood": "High/Medium/Low", "reason": "Reason"}],
    "information_gaps": ["List of missing information"],
    "retrieval_topics": ["List of topics to retrieve"],
    "revised_diagnosis": "Revised diagnosis (if any)",
    "confidence_change": "Increased/Decreased/Unchanged"
}
```
"""

RETRIEVAL_PROMPT = """
Based on the following medical question, retrieve relevant medical knowledge:

{question}

Please retrieve medical knowledge related to the following topics:
{topics}

Please return the most relevant medical knowledge content and explain how it helps your diagnosis.
"""

FINAL_DIAGNOSIS_PROMPT = """
### Task
Based on all previous rounds of analysis and discussion, provide the final diagnosis.

### Considerations
1. Initial diagnosis analysis
2. Critical review results
3. Retrieved medical knowledge
4. All available clinical information

### Output Requirements
```json
{
    "final_diagnosis": "Final diagnosis name",
    "diagnosis_basis": ["List of diagnostic basis"],
    "confidence": 0.95,
    "reasoning_summary": "Summary of reasoning process",
    "alternative_diagnoses": ["Other diagnoses to rule out"]
}
```

Finally, please directly output the final diagnosis (only the diagnosis name, no other descriptions):
"""

SELF_DIALOGUE_PROMPT_WITH_OPTIONS = """
### Role Setting
You are a rigorous medical expert conducting self-diagnosis and reflection on a clinical case.

### Task Requirements
You need to complete the diagnostic task through multiple rounds of self-dialogue. Multiple choice options are provided - you must select the correct answer from these options.

### Multiple Choice Options
{options}

Each round must complete the following steps:

## Round 1: Initial Diagnosis
Based on the case description and imaging data, select the most likely diagnosis from the options above.

## Round 2: Critical Review
Switch to a critical perspective and review your initial selection:
1. What is the most important supporting evidence for your choice?
2. Which other options are most likely to be confused with your choice?
3. What information would help confirm your selection?

## Round 3: Knowledge Retrieval and Verification
Based on the questions from the previous round, retrieve relevant medical knowledge to verify your choice.

## Round 4: Final Confirmation
Synthesize all information and provide your final selection from the options.

### Output Format
After each round, please output in the following format:

```json
{{
    "round": 1,
    "type": "initial_diagnosis/preliminary_criticism/knowledge_retrieval/final_confirmation",
    "reasoning": "Your analytical reasoning process",
    "selected_option": "The option letter or text you select",
    "confidence": 0.8,
    "self_questions": ["List of questions for self-inquiry"],
    "needs_retrieval": true/false
}}
```

### Important Notes
1. You MUST select from the provided options
2. Honestly assess your diagnostic confidence in each round
3. When identifying need for more information, clearly state retrieval requirements
4. Final answer must be one of the provided options

Please start Round 1.
"""

SELF_CRITICAL_PROMPT_WITH_OPTIONS = """
### Task
Please critically review your initial diagnosis selection from the options.

### Analysis Dimensions
1. **Supporting Evidence Analysis**: List evidence supporting your selected option
2. **Option Comparison**: Compare your selection with other likely options
3. **Information Gaps**: What additional information would confirm your selection?
4. **Retrieval Needs**: What medical knowledge needs to be verified?

### Output Requirements
Please output in JSON format:
```json
{
    "supporting_evidence": ["Evidence for your selection"],
    "option_comparisons": [{"option": "Option text", "likelihood": "High/Medium/Low", "reason": "Comparison reason"}],
    "information_gaps": ["Missing information"],
    "retrieval_topics": ["Topics to verify"],
    "revised_selection": "Revised option (if any)",
    "confidence_change": "Increased/Decreased/Unchanged"
}
```
"""

FINAL_DIAGNOSIS_PROMPT_WITH_OPTIONS = """
### Task
Based on all previous rounds of analysis and discussion, provide your final selection from the options.

### Considerations
1. Initial diagnosis analysis
2. Critical review results
3. Retrieved medical knowledge verification
4. Option-by-option comparison

### Output Requirements
```json
{
    "final_selection": "The exact option text you select",
    "diagnosis_basis": ["Reasons for selection"],
    "confidence": 0.95,
    "reasoning_summary": "Brief reasoning summary"
}
```

Finally, please directly output the option text (exactly as it appears in the options list):
"""


def load_knowledge_base(embeddings_model):
    """加载知识库"""
    try:
        if os.path.exists(os.path.join(knowledge_path, "index.faiss")):
            db = FAISS.load_local(knowledge_path, embeddings=embeddings_model)
            return db
    except Exception as e:
        logger.error(f"加载知识库Failed: {e}")
    return None


def retrieve_knowledge(db, query, k=3):
    """从知识库检索相关知识"""
    if db is None:
        return []
    try:
        docs = db.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    except Exception as e:
        logger.error(f"知识检索Failed: {e}")
        return []


def create_self_dialogue_messages(rag_context, conversation_history, current_round, round_type):
    """创建多轮对话的消息"""
    messages = [{"role": "system", "content": SELF_DIALOGUE_PROMPT}]

    if rag_context:
        context_str = "\n\n### 参考知识\n" + "\n---\n".join(rag_context)
        messages.append({
            "role": "user",
            "content": f"{context_str}\n\n{conversation_history}\n\n请继续第{current_round}轮对话（{round_type}）。"
        })
    else:
        messages.append({
            "role": "user",
            "content": f"{conversation_history}\n\n请继续第{current_round}轮对话（{round_type}）。"
        })

    return messages


def parse_round_output(output):
    """解析每一轮的输出"""
    try:
        if not output or not output.strip():
            return None

        json_str = output

        if "```json" in output:
            json_str = output.split("```json")[1].split("```")[0]
        elif "```" in output:
            json_str = output.split("```")[1].split("```")[0]

        json_str = json_str.strip()

        json_str = json_str.replace('\n    ', ' ').replace('\n', ' ')

        json_str = json_str.replace('    ', ' ')

        parsed = json.loads(json_str)

        if parsed and isinstance(parsed, dict):
            return parsed
        else:
            return None

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}, attempting to clean and retry...")
        try:
            json_str = output
            if "```json" in output:
                json_str = output.split("```json")[1].split("```")[0]
            elif "```" in output:
                json_str = output.split("```")[1].split("```")[0]

            lines = json_str.split('\n')
            cleaned_lines = []
            for line in lines:
                if '"round"' not in line and '"diagnosis"' not in line:
                    cleaned_lines.append(line)

            json_str = '\n'.join(cleaned_lines)

            json_str = re.sub(r'\s+', ' ', json_str)

            json_str = re.sub(r'\s*([{}\[\]:,])\s*', r'\1', json_str)

            parsed = json.loads(json_str)
            if parsed and isinstance(parsed, dict):
                return parsed
        except Exception as e2:
            logger.error(f"Failed to parse even after cleaning: {e2}")
        return None
    except Exception as e:
        logger.error(f"Parse round output failed: {e}")
        return None


def extract_final_diagnosis(output):
    """Extract final diagnosis from output"""
    try:
        if "Final Diagnosis:" in output:
            diagnosis = output.split("Final Diagnosis:")[-1].strip()
            if "\n" in diagnosis:
                diagnosis = diagnosis.split("\n")[0].strip()
            return diagnosis
        if "final_diagnosis" in output.lower():
            import re
            match = re.search(r'["\']?final_diagnosis["\']?\s*[:：]\s*["\']([^"\']+)["\']?', output)
            if match:
                return match.group(1).strip()
        lines = output.strip().split("\n")
        for line in lines:
            if line and not line.startswith("```") and len(line) < 100:
                if any(char in line for char in "，。、；：？！"):
                    return line.strip()
        return output.strip()[:100] if output.strip() else "Unable to parse diagnosis result"
    except Exception as e:
        logger.error(f"Failed to extract final diagnosis: {e}")
        return "Unable to parse diagnosis result"


def extract_option_from_output(output, options_text):
    """Extract selected option from output"""
    try:
        if "final_selection" in output.lower():
            import re
            match = re.search(r'["\']?final_selection["\']?\s*[:：]\s*["\']([^"\']+)["\']?', output, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        if "selected_option" in output.lower():
            import re
            match = re.search(r'["\']?selected_option["\']?\s*[:：]\s*["\']([^"\']+)["\']?', output, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        option_lines = options_text.split('\n')
        for line in option_lines:
            line = line.strip()
            if line and line in output:
                return line

        if "Option:" in output or "option:" in output:
            for line in output.split('\n'):
                if 'Option' in line or 'option' in line:
                    return line.split(':')[-1].strip()

        return output.strip()[:200] if output.strip() else "Unable to parse option"
    except Exception as e:
        logger.error(f"Failed to extract option: {e}")
        return "Unable to parse option"


def run_self_dialogue(model_name, content, image_path, target, openai_client, encode_image_func, db, options=None):
    """
    Execute multi-round self-dialogue

    Args:
        model_name: Model name
        content: Clinical description
        image_path: Path to image
        target: Target diagnosis
        openai_client: OpenAI client
        encode_image_func: Image encoding function
        db: Knowledge base
        options: Multiple choice options (if available)

    Returns:
        dict: Contains dialogue history and final diagnosis
    """
    try:
        image_base64 = encode_image_func(image_path)

        use_options = options is not None and str(options).strip() != ""
        options_text = str(options).strip() if options else ""

        conversation_history = f"""
### Case Information
**Clinical Description**: {content}

**Imaging Data**: (See attached image)
"""

        if use_options:
            conversation_history += f"""
### Multiple Choice Options
{options_text}

Please select the correct answer from the options above.
"""
        else:
            conversation_history += """
Please provide your diagnosis based on the clinical information.
"""

        if use_options:
            system_prompt = SELF_DIALOGUE_PROMPT_WITH_OPTIONS.format(options=options_text)
        else:
            system_prompt = SELF_DIALOGUE_PROMPT

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpg;base64,{image_base64}"
                        },
                    },
                    {"type": "text", "text": conversation_history},
                ],
            },
        ]

        round_outputs = []
        final_diagnosis = None
        max_rounds = 4

        for round_num in range(1, max_rounds + 1):
            if should_exit or exit_event.is_set():
                break

            output = model_process(openai_client, messages, model_name)
            if not output:
                break

            parsed = parse_round_output(output)
            round_outputs.append({
                "round": round_num,
                "raw_output": output,
                "parsed": parsed
            })

            if parsed:
                diagnosis = parsed.get("diagnosis", "")
                needs_retrieval = parsed.get("needs_retrieval", False)
                self_questions = parsed.get("self_questions", [])

                rag_context = []
                if needs_retrieval and db:
                    topics = parsed.get("retrieval_topics", self_questions)
                    query = f"{content} {diagnosis} {' '.join(topics[:3])}" if topics else f"{content} {diagnosis}"
                    rag_docs = retrieve_knowledge(db, query, k=3)
                    rag_context = rag_docs

                    retrieval_output = ""
                    if rag_docs:
                        retrieval_prompt = RETRIEVAL_PROMPT.format(
                            question=f"病例诊断问题：{content}",
                            topics="\n".join([f"- {t}" for t in topics[:3]])
                        )
                        retrieval_messages = [{"role": "user", "content": retrieval_prompt}]
                        retrieval_output = model_process(openai_client, retrieval_messages, model_name)
                        if retrieval_output:
                            rag_context.append(f"\n### 检索分析\n{retrieval_output}")
            else:
                logger.warning(f"Round {round_num}: Failed to parse model output, using fallback")
                diagnosis = ""
                needs_retrieval = True
                self_questions = []
                rag_context = []

            if round_num < max_rounds:
                if round_num == 1:
                    next_prompt = SELF_CRITICAL_PROMPT_WITH_OPTIONS if use_options else SELF_CRITICAL_PROMPT
                elif round_num == 2:
                    if use_options:
                        next_prompt = RETRIEVAL_PROMPT.format(
                            question=f"Diagnostic question: {content}",
                            topics="\n".join([f"- {t}" for t in (parsed.get("retrieval_topics", []) if parsed else [])])
                        ) if (parsed and parsed.get(
                            "retrieval_topics")) else "Based on your critical review, provide further diagnostic analysis."
                    else:
                        next_prompt = RETRIEVAL_PROMPT.format(
                            question=f"Diagnostic question: {content}",
                            topics="\n".join([f"- {t}" for t in (parsed.get("retrieval_topics", []) if parsed else [])])
                        ) if (parsed and parsed.get(
                            "retrieval_topics")) else "Based on your critical review, provide further diagnostic analysis."
                else:
                    next_prompt = FINAL_DIAGNOSIS_PROMPT_WITH_OPTIONS if use_options else FINAL_DIAGNOSIS_PROMPT

                context_msg = ""
                if rag_context:
                    context_msg = f"\n\n### Reference Knowledge\n" + "\n---\n".join(rag_context)

                messages.append({"role": "assistant", "content": output})
                messages.append({
                    "role": "user",
                    "content": f"{context_msg}\n\nPlease continue with {next_prompt}"
                })
            else:
                if use_options:
                    final_diagnosis = extract_option_from_output(output, options_text)
                else:
                    final_diagnosis = extract_final_diagnosis(output)

        if not final_diagnosis:
            if round_outputs:
                last_parsed = round_outputs[-1].get("parsed", {})
                final_diagnosis = last_parsed.get("diagnosis", "Unable to determine diagnosis")
            else:
                final_diagnosis = "Unable to determine diagnosis"

        return {
            "conversation_history": conversation_history,
            "round_outputs": round_outputs,
            "final_diagnosis": final_diagnosis,
            "num_rounds": len(round_outputs)
        }

    except Exception as e:
        logger.error(f"多轮对话执行Failed: {e}")
        return {
            "error": str(e),
            "final_diagnosis": f"ERROR: {str(e)}"
        }


def create_task_id(model_name, data_index, content):
    """创建任务唯一ID"""
    task_str = f"{model_name}_{data_index}_{content[:100]}"
    return hashlib.md5(task_str.encode()).hexdigest()[:12]


def evaluate_multiround_wrapper(model_name, content, image_path, target, task_id, openai_client,
                                encode_image_func, db, model_name_list, options=None):
    """
    Wrapper for multi-round evaluation function, for concurrent calls

    Args:
        model_name: Model name
        content: Clinical description
        image_path: Image path
        target: Target diagnosis
        task_id: Task ID
        openai_client: OpenAI client
        encode_image_func: Image encoding function
        db: Knowledge base
        model_name_list: Model names for diagnostic judgment
        options: Multiple choice options (if available)
    """
    try:
        use_options = options is not None and str(options).strip() != ""

        result = run_self_dialogue(
            model_name=model_name,
            content=content,
            image_path=image_path,
            target=target,
            openai_client=openai_client,
            encode_image_func=encode_image_func,
            db=db,
            options=options
        )

        if "error" in result:
            return {
                "task_id": task_id,
                "model_name": model_name,
                "input": content,
                "image_path": image_path,
                "output": result.get("final_diagnosis", f"ERROR: {result['error']}"),
                "target": target,
                "flag": False,
                "status": "failed",
                "error": result.get("error"),
                "conversation_rounds": 0,
                "use_options": use_options,
                "options": str(options) if use_options else "",
                "timestamp": datetime.now().isoformat()
            }

        output_diagnosis = result["final_diagnosis"]
        flag = diagnostic_judge(openai_client, target, output_diagnosis,
                                model_name_list[0] if model_name_list else "gpt-5-mini")

        # conversation_json = json.dumps(result["round_outputs"], ensure_ascii=False, indent=2)

        return {
            "task_id": task_id,
            "model_name": model_name,
            "input": content,
            "image_path": image_path,
            "output": output_diagnosis,
            "target": target,
            "flag": bool(flag),
            "status": "success",
            "conversation_rounds": result["num_rounds"],
            # "conversation_history": conversation_json,
            "use_options": use_options,
            "options": str(options) if use_options else "",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Model {model_name} multi-round dialogue processing failed: {str(e)}")
        return {
            "task_id": task_id,
            "model_name": model_name,
            "input": content,
            "image_path": image_path,
            "output": f"ERROR: {str(e)}",
            "target": target,
            "flag": False,
            "status": "failed",
            "error": str(e),
            "conversation_rounds": 0,
            "use_options": options is not None and str(options).strip() != "",
            "options": str(options) if options else "",
            "timestamp": datetime.now().isoformat()
        }


def get_completed_task_ids_from_file(result_file):
    """从结果文件中获取已完成的任务ID"""
    completed_ids = set()

    if os.path.exists(result_file):
        try:
            existing_df = pd.read_csv(result_file)
            if not existing_df.empty and 'task_id' in existing_df.columns:
                completed_ids = set(existing_df['task_id'].dropna().astype(str).tolist())
                logger.info(f"从结果文件加载了 {len(completed_ids)} 个已完成任务")
        except Exception as e:
            logger.warning(f"读取结果文件Failed: {str(e)}，将重新开始")

    return completed_ids


def process_multiround_parallel(openai_client, train_data, model_names, output_dir,
                                encode_image_func, model_name_list,
                                max_workers=3, checkpoint_interval=5, has_options=True):
    """
    Concurrent processing of multi-round dialogue evaluation tasks

    Args:
        openai_client: OpenAI client
        train_data: Training data
        model_names: Model name list
        output_dir: Output directory
        encode_image_func: Image encoding function
        model_name_list: Model names for diagnostic judgment
        max_workers: Maximum concurrent workers
        checkpoint_interval: Checkpoint save interval (number of tasks)
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_file = os.path.join(output_dir, "multiround_results.csv")
    temp_result_file = os.path.join(output_dir, "multiround_results_temp.csv")

    db = load_knowledge_base(embeddings_model)
    if db:
        logger.info("Knowledge base loaded successfully")
    else:
        logger.warning("Knowledge base not found, RAG enhancement will not be used")

    completed_task_ids = get_completed_task_ids_from_file(temp_result_file)

    all_tasks = []
    for i in range(len(train_data)):
        content = train_data["input"][i]
        target = train_data["target"][i]
        image_path = train_data["image_path"][i]
        options = train_data["options"][i] if has_options else None
        for model_name in model_names:
            task_id = create_task_id(model_name, i, content)
            all_tasks.append({
                "task_id": task_id,
                "model_name": model_name,
                "content": content,
                "image_path": image_path,
                "target": target,
                "options": options,
                "data_index": i
            })

    pending_tasks = [task for task in all_tasks if task["task_id"] not in completed_task_ids]

    logger.info(f"总任务数: {len(all_tasks)}, 已完成: {len(completed_task_ids)}, 待处理: {len(pending_tasks)}")

    if len(pending_tasks) == 0:
        logger.info("所有任务已完成，跳过处理")
        if os.path.exists(result_file):
            return pd.read_csv(result_file)
        return pd.DataFrame()

    if os.path.exists(temp_result_file) and len(completed_task_ids) > 0:
        result_df = pd.read_csv(temp_result_file)
    else:
        result_df = pd.DataFrame(columns=[
            "task_id", "model_name", "input", "image_path", "output",
            "target", "flag", "status", "conversation_rounds", "timestamp", "error", "use_options", "options"
        ])

    pbar = tqdm(total=len(pending_tasks), desc="Multi-round dialogue processing progress")
    completed_count = len(completed_task_ids)
    lock = threading.Lock()

    evaluate_func = partial(
        evaluate_multiround_wrapper,
        openai_client=openai_client,
        encode_image_func=encode_image_func,
        db=db,
        model_name_list=model_name_list
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
                        logger.debug(f"临时结果已保存: {temp_result_file} ({completed_count} 个任务)")
                    except Exception as e:
                        logger.error(f"保存临时结果Failed: {str(e)}")

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

    def signal_handler(signum, frame):
        global should_exit
        logger.info("Received interrupt signal, saving progress...")
        should_exit = True
        exit_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for task_detail in pending_tasks:
            if should_exit or exit_event.is_set():
                logger.info("Exit signal detected, stopping task submission")
                break

            future = executor.submit(
                evaluate_func,
                model_name=task_detail["model_name"],
                content=task_detail["content"],
                image_path=task_detail["image_path"],
                target=task_detail["target"],
                task_id=task_detail["task_id"],
                options=task_detail.get("options")
            )

            future.add_done_callback(
                lambda f, td=task_detail: task_callback(f, td)
            )

            futures.append(future)

            if len(futures) >= max_workers * 2:
                concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                futures = [f for f in futures if not f.done()]

        try:
            if not should_exit and not exit_event.is_set():
                concurrent.futures.wait(futures)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, waiting for tasks to complete...")
            for future in futures:
                try:
                    future.result(timeout=2)
                except:
                    pass

    pbar.close()

    try:
        if not result_df.empty:
            result_df.to_csv(result_file, index=False, encoding='utf-8')
            logger.info(f"Final results saved to: {result_file}")

            if os.path.exists(temp_result_file):
                os.remove(temp_result_file)
                logger.debug(f"Temporary file deleted: {temp_result_file}")

            success_count = len(result_df[result_df['status'] == 'success'])
            failed_count = len(result_df[result_df['status'] == 'failed'])
            logger.info(f"Processing complete! Success: {success_count}, Failed: {failed_count}")

    except Exception as e:
        logger.error(f"保存最终结果Failed: {str(e)}")
        if os.path.exists(temp_result_file):
            os.rename(temp_result_file, result_file)
            logger.info(f"Results recovered from temp file: {result_file}")

    return result_df


if __name__ == '__main__':
    train_data = pd.read_csv("./data/train_with_options.csv")

    # 控制是否带选项试验
    # has_options = True
    has_options = False

    dataset_name = "with_options" if has_options else "no_options"
    output_dir = f"./output/multiround_eval_{dataset_name}"

    os.makedirs(output_dir, exist_ok=True)

    max_workers = 2
    checkpoint_interval = 3

    model_names = ["claude-opus-4-5-20251101", "gpt-4o", "gemini-3-pro-preview"]
    model_name_list = ["gpt-5-mini"]

    print(f"Running multi-round dialogue test - Mode: {dataset_name}")
    print(f"Output directory: {output_dir}")
    print(f"Total samples: {len(train_data)}")
    print(f"Has options: {has_options}")
    print("-" * 60)

    result_df = process_multiround_parallel(
        openai_client=openai_client,
        train_data=train_data,
        model_names=model_names,
        output_dir=output_dir,
        encode_image_func=encode_image,
        model_name_list=model_name_list,
        max_workers=max_workers,
        checkpoint_interval=checkpoint_interval,
        has_options=has_options
    )

    print("\n" + "=" * 60)
    print("Multi-Round Dialogue Test Complete!")
    print("=" * 60)
