import os
import sys
import argparse
import pandas as pd
from openai import OpenAI
from loguru import logger
from datetime import datetime
import threading
import time
from config import openai_client

# 配置日志
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

# 全局退出控制
should_exit = False
exit_event = threading.Event()


# 监听 Ctrl+C
def signal_handler(signum, frame):
    global should_exit
    should_exit = True
    exit_event.set()
    logger.warning("收到中断信号，正在保存进度并退出...")


import signal

signal.signal(signal.SIGINT, signal_handler)

# 角色定义（来自 role_experiment.py）
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

# 角色配置：固定的角色分配
roles = [
    "",  # 主管理模型
    "Imaging Department + Laboratory Department",  # 影像科 + 检验科
    "Pathology Department + Oncology Department"  # 病理科 + 肿瘤科
]

ROLE_NAMES = {
    "": "主管理模型",
    "Imaging Department + Laboratory Department": "影像科 + 检验科",
    "Pathology Department + Oncology Department": "病理科 + 肿瘤科"
}

# 固定的角色分配方案
FIXED_ROLE_ASSIGNMENT = {
    "claude": "Pathology Department + Oncology Department",  # 病理科 + 肿瘤科
    "gemini": "",  # 主管理模型
    "deepseek": "Imaging Department + Laboratory Department"  # 影像科 + 检验科
}

# 模型配置
models = [
    {
        "name": "claude",
        "model": "claude-opus-4-5-20251101",
        "has_vision": True  # Claude 原生支持视觉
    },
    {
        "name": "gemini",
        "model": "gemini-3-pro-preview",
        "has_vision": True  # Gemini 原生支持视觉
    },
    {
        "name": "deepseek",
        "model": "deepseek-v3-1-think-250821",
        "has_vision": False,  # DeepSeek 不支持视觉
        "assist_model": "claude-opus-4-5-20251101"  # 使用 Claude 描述图像
    }
]

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


def describe_image_with_model(image_base64, model_id, client):
    """使用指定模型描述图像"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
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
                        {
                            "type": "text",
                            "text": "Please provide a detailed description of this medical image. Focus on visible anatomical structures and any abnormalities."
                        },
                    ]
                }
            ]
            
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=1000,
                timeout=30
            )
            
            # 验证响应是否有效
            if not response or not response.choices or len(response.choices) == 0:
                logger.warning(f"Empty response from {model_id}, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return ""
            
            content = response.choices[0].message.content
            
            # 检查内容是否有效（不是 HTML 错误页面或其他无效内容）
            if content and (content.startswith('<') or 'error' in content.lower()):
                logger.warning(f"Invalid response content from {model_id}, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
            
            return content if content else ""
            
        except Exception as e:
            logger.warning(f"Image description failed for {model_id} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                logger.error(f"All {max_retries} attempts failed for image description")
                return ""
    
    return ""


def parse_model_output(output):
    """解析模型输出，提取诊断结果"""
    if not output:
        return None

    try:
        output = output.strip()
        lines = output.split('\n')

        diagnosis = None
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                diagnosis = line
                break

        if not diagnosis:
            diagnosis = output

        return {
            "diagnosis": diagnosis,
        }
    except Exception as e:
        logger.error(f"Failed to parse output: {e}")
        return None


def run_collaborative_diagnosis(content, image_path, target, openai_client, encode_image_func, options=None,
                                use_options=True):
    """
    运行多轮协作诊断
    """
    try:
        options_text = str(options).strip() if options else ""
        image_base64 = encode_image_func(image_path)

        # 使用外部角色 prompt（如果有）
        external_role = ""
        base_role_prompt = DEFAULT_PROMPT if external_role == "" else ROLE_PROMPT.replace("role_template",
                                                                                          external_role)

        discussion_history = []
        model_opinions = []
        max_rounds = 3

        # 根据是否使用选项来构建 prompt
        if use_options and options_text:
            options_section = f"""
### Multiple Choice Options
{options_text}

Please select one option from the above choices."""
        else:
            options_section = ""

        for round_num in range(1, max_rounds + 1):
            if should_exit or exit_event.is_set():
                break

            round_outputs = []

            for model_info in models:
                model_name = model_info["name"]
                model_id = model_info["model"]
                assigned_role = FIXED_ROLE_ASSIGNMENT[model_name]
                has_vision = model_info.get("has_vision", False)
                assist_model = model_info.get("assist_model")
                
                # 构建模型的系统 prompt
                if assigned_role == "":
                    # 主管理模型，使用 DEFAULT_PROMPT
                    system_prompt = base_role_prompt
                else:
                    # 科室角色，使用 ROLE_PROMPT
                    system_prompt = base_role_prompt + "\n\n" + ROLE_PROMPT.replace("role_template", assigned_role)
                
                # 处理图像：只有不支持视觉的模型才需要助理模型描述
                image_description = ""
                if not has_vision and assist_model:
                    # DeepSeek 等不支持视觉的模型，使用助理模型描述图像
                    logger.info(f"Using {assist_model} to describe image for {model_name}")
                    image_description = describe_image_with_model(image_base64, assist_model, openai_client)
                
                if image_description:
                    imaging_data_text = f"**Image Description**:\n{image_description}"
                else:
                    imaging_data_text = "(See attached image)"
                
                init_prompt = f"""### Case Information
**Clinical Description**: {content}

**Imaging Data**: {imaging_data_text}
{options_section}

### Task
You are part of a multi-disciplinary team. Please introduce yourself and share your initial assessment based on your expertise."""
                
                # 构建消息
                if has_vision:
                    # Claude 和 Gemini 直接处理图像
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
                                {"type": "text", "text": init_prompt},
                            ]
                        }
                    ]
                else:
                    # DeepSeek 等使用文本 + 图像描述
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": init_prompt}
                    ]

                if not image_description:
                    # 只有当没有图像描述时（即支持视觉的模型），才需要添加图像
                    # 但实际上支持视觉的模型已经在上面添加了图像，所以这里不需要再处理
                    pass
                
                if round_num > 1:
                    discussion_text = "\n\n".join([
                        f"### {ROLE_NAMES[op['role']]} - {op['model']} (Round {op['round']})\n{op['raw_output']}"
                        for op in discussion_history
                    ])
                    
                    next_prompt = f"""### Previous Discussion
{discussion_text}

### Task
Based on the discussion above, please provide your updated assessment. Consider other specialists' opinions and explain how they influence your diagnosis.

If you agree with a specific diagnosis from the options, clearly state your agreement and reasoning."""
                    
                    if has_vision:
                        # 支持视觉的模型在第二轮及以后添加历史对话
                        messages.append({"role": "assistant", "content": "Previous response incorporated"})
                        messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpg;base64,{image_base64}"
                                    },
                                },
                                {"type": "text", "text": next_prompt},
                            ]
                        })
                    else:
                        # 不支持视觉的模型只添加文本
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
            logger.info(f"Round {round_num} completed")

        # Final consensus round
        discussion_text = "\n\n".join([
            f"### {op['role_name']} - {op['model']} (Round {op['round']})\n{op['raw_output']}"
            for op in discussion_history
        ])

        if use_options and options_text:
            options_instruction = " Select one option from the multiple choice options above and explain your reasoning."
        else:
            options_instruction = ""

        final_consensus_prompt_text = f"""### Complete Discussion History
{discussion_text}

### Task
Based on the entire discussion, please provide your final diagnostic conclusion.{options_instruction}

If there is disagreement among specialists, acknowledge it but still provide your best assessment."""

        final_opinions = []
        for model_info in models:
            model_name = model_info["name"]
            model_id = model_info["model"]
            assigned_role = FIXED_ROLE_ASSIGNMENT[model_name]
            has_vision = model_info.get("has_vision", False)
            
            if assigned_role == "":
                system_prompt = base_role_prompt
            else:
                system_prompt = base_role_prompt + "\n\n" + ROLE_PROMPT.replace("role_template", assigned_role)
            
            # 构建最终共识的消息
            if has_vision:
                # 支持视觉的模型直接添加图像
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
                            {"type": "text", "text": final_consensus_prompt_text},
                        ]
                    }
                ]
            else:
                # 不支持视觉的模型只使用文本
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": final_consensus_prompt_text}
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


def model_process(model_client, messages, model_name="gpt-5-chat-latest"):
    try:
        response = model_client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(e)
    return None


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        import base64
        return base64.b64encode(image_file.read()).decode("utf-8")


def process_single_case(openai_client, train_data, index, output_dir, encode_image_func, use_options=True):
    """处理单个病例"""
    try:
        content = train_data["input"].iloc[index]
        target = train_data["target"].iloc[index]
        image_path = train_data["image_path"].iloc[index]
        options = train_data["options"].iloc[index] if "options" in train_data.columns else None

        logger.info(f"Processing case {index + 1}/{len(train_data)}")

        result = run_collaborative_diagnosis(
            content=content,
            image_path=image_path,
            target=target,
            openai_client=openai_client,
            encode_image_func=encode_image_func,
            options=options,
            use_options=use_options
        )

        if "error" in result:
            logger.error(f"Case {index + 1} failed: {result['error']}")
            return None

        # 保存结果
        results = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for final_op in result["final_opinions"]:
            result_row = {
                "model_name": final_op["model"],
                "model_id": final_op["model_id"],
                "assigned_role": final_op["role"],
                "role_name": final_op["role_name"],
                "role_assignment_id": "|".join(
                    [f"{m}={ROLE_NAMES[FIXED_ROLE_ASSIGNMENT[m]]}" for m in ["claude", "gemini", "deepseek"]]),
                "input": content,
                "image_path": image_path,
                "output": final_op["final_diagnosis"],
                "target": target,
                "status": "success",
                "conversation_rounds": result["num_rounds"],
                "timestamp": timestamp,
                "error": "",
                "use_options": use_options
            }
            results.append(result_row)

        # 保存讨论历史
        discussion_file = os.path.join(output_dir, f"case_{index + 1:04d}_discussion.txt")
        with open(discussion_file, "w", encoding="utf-8") as f:
            f.write(f"Case: {content[:200]}...\n")
            f.write(f"Target: {target}\n")
            f.write(f"Options: {options}\n\n")
            f.write("=" * 80 + "\n\n")

            for op in result["discussion_history"]:
                f.write(f"### {op['role_name']} - {op['model']} (Round {op['round']})\n")
                f.write(f"{op['raw_output']}\n\n")
                f.write("-" * 80 + "\n\n")

        return results

    except Exception as e:
        logger.error(f"Processing case {index + 1} failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="多轮协作诊断最终版实验")
    parser.add_argument("--with-options", action="store_true", help="是否将选项传给模型")
    parser.add_argument("--start-index", type=int, default=0, help="起始索引")
    parser.add_argument("--end-index", type=int, default=None, help="结束索引（不包含）")
    parser.add_argument("--output-dir", type=str, default="./output/multiround_final", help="输出目录")

    args = parser.parse_args()

    # 创建输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    data_file = "./data/train_with_options.csv"
    if not os.path.exists(data_file):
        logger.error(f"数据文件不存在：{data_file}")
        return

    train_data = pd.read_csv(data_file)
    train_data = train_data.iloc[:2, :]
    train_data.reset_index(drop=True, inplace=True)

    # 应用索引范围
    start_index = args.start_index
    end_index = args.end_index if args.end_index is not None else len(train_data)
    train_data = train_data.iloc[start_index:end_index, :]
    train_data.reset_index(drop=True, inplace=True)

    # 打印配置信息
    print("\n" + "=" * 80)
    print("多轮协作诊断最终版实验")
    print("=" * 80)
    print(f"输出目录：{output_dir}")
    print(f"总样本数：{len(train_data)}")
    print(f"模型：{[m['name'] for m in models]}")
    print(f"角色配置:")
    for model_name, role in FIXED_ROLE_ASSIGNMENT.items():
        print(f"  {model_name}: {ROLE_NAMES[role]}")
    print(f"是否使用选项：{'是' if args.with_options else '否'}")
    print(f"索引范围：{start_index} - {end_index}")
    print("=" * 80 + "\n")

    # 处理病例
    all_results = []
    for i in range(len(train_data)):
        if should_exit or exit_event.is_set():
            logger.warning("收到中断信号，停止处理")
            break

        results = process_single_case(
            openai_client=openai_client,
            train_data=train_data,
            index=i,
            output_dir=output_dir,
            encode_image_func=encode_image,
            use_options=args.with_options
        )

        if results:
            all_results.extend(results)

            # 每处理 5 个病例保存一次
            if (i + 1) % 5 == 0:
                result_df = pd.DataFrame(all_results)
                result_file = os.path.join(output_dir, "final_results.csv")
                result_df.to_csv(result_file, index=False, encoding="utf-8-sig")
                logger.info(f"已保存 {len(all_results)} 条结果到 {result_file}")

    # 保存最终结果
    if all_results:
        result_df = pd.DataFrame(all_results)
        result_file = os.path.join(output_dir, "final_results.csv")
        result_df.to_csv(result_file, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 80)
        print("处理完成!")
        print(f"总记录数：{len(result_df)}")
        print(f"输出文件：{result_file}")
        print("=" * 80)
    else:
        logger.warning("没有生成任何结果")


if __name__ == "__main__":
    main()
