"""
-*- coding: utf-8 -*-

@Author  : houcg
@Time    : 2026/1/8 11:11
"""
import os

import pandas
from config import openai_client
from loguru import logger
from utils.llm_util import encode_image, model_process, diagnostic_judge
import os
import pandas as pd
import concurrent.futures
import threading
from tqdm import tqdm
import json
from datetime import datetime
from functools import partial
import signal

# 全局变量用于优雅退出
should_exit = False
exit_event = threading.Event()

"""
第一轮：
先选定部分数据进行3个模型的训练，找出最好的模型，作为结果1
"""

model_names = ["claude-opus-4-5-20251101", "gpt-5.2", "gemini-3-pro-preview"]
train_data = pandas.read_csv("./data/train.csv")

PROMPT = """
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


def signal_handler(signum, frame):
    """处理中断信号"""
    global should_exit
    logger.info("收到中断信号，正在保存当前进度...")
    should_exit = True
    exit_event.set()


# 注册信号处理
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def create_task_id(model_name, data_index, content):
    """创建任务唯一ID"""
    # 使用简单的哈希值作为任务ID
    import hashlib
    task_str = f"{model_name}_{data_index}_{content[:100]}"
    return hashlib.md5(task_str.encode()).hexdigest()[:12]


def evaluate_model_wrapper(model_name, content, image_path, target, task_id, openai_client, PROMPT,
                           model_process, diagnostic_judge, encode_image):
    """包装评估函数，便于并发调用"""
    try:
        image_base64 = encode_image(image_path)
        messages = [
            {"role": "system", "content": PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpg;base64,{image_base64}"
                        },
                    },
                    {"type": "text", "text": content},
                ],
            },
        ]

        output = model_process(openai_client, messages, model_name)
        flag = diagnostic_judge(openai_client, target, output)

        logger.debug(f"model: {model_name}, output: {output}, target: {target}, flag: {flag}")

        if flag:
            return {
                "task_id": task_id,
                "model_name": model_name,
                "input": content,
                "image_path": image_path,
                "output": target,  # 当flag为True时，返回target
                "target": target,
                "flag": flag,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "task_id": task_id,
                "model_name": model_name,
                "input": content,
                "image_path": image_path,
                "output": output,
                "target": target,
                "flag": flag,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"模型 {model_name} 处理失败: {str(e)}")
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
            "timestamp": datetime.now().isoformat()
        }


def get_completed_task_ids_from_file(result_file):
    """从结果文件中获取已完成的任务ID"""
    completed_ids = set()

    if os.path.exists(result_file):
        try:
            # 读取已有结果
            existing_df = pd.read_csv(result_file)
            if not existing_df.empty and 'task_id' in existing_df.columns:
                completed_ids = set(existing_df['task_id'].dropna().astype(str).tolist())
                logger.info(f"从结果文件加载了 {len(completed_ids)} 个已完成任务")
        except Exception as e:
            logger.warning(f"读取结果文件失败: {str(e)}，将重新开始")

    return completed_ids


def save_progress_info(output_dir, total_tasks, completed_tasks, failed_tasks):
    """保存进度信息到独立文件"""
    progress_file = os.path.join(output_dir, "progress.json")

    progress_data = {
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "failed_tasks": failed_tasks,
        "timestamp": datetime.now().isoformat(),
        "progress_percentage": (completed_tasks + failed_tasks) / total_tasks * 100 if total_tasks > 0 else 0
    }

    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"保存进度信息失败: {str(e)}")


def process_batch_parallel(openai_client, train_data, model_names, output_dir,
                           model_process, diagnostic_judge, encode_image, PROMPT,
                           max_workers=5, checkpoint_interval=5):
    """
    并发处理模型评估任务

    Args:
        openai_client: OpenAI客户端
        train_data: 训练数据
        model_names: 模型名称列表
        output_dir: 输出目录
        model_process: 模型处理函数
        diagnostic_judge: 诊断判断函数
        encode_image: 图片编码函数
        PROMPT: 系统提示词
        max_workers: 最大并发数
        checkpoint_interval: 检查点保存间隔（任务数）
    """

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 结果文件路径
    result_file = os.path.join(output_dir, "model_results.csv")
    temp_result_file = os.path.join(output_dir, "model_results_temp.csv")

    # 获取已完成的任务ID
    completed_task_ids = get_completed_task_ids_from_file(temp_result_file)

    # 准备所有任务
    all_tasks = []

    for i in range(len(train_data)):
        content = train_data["input"][i]
        target = train_data["target"][i]
        image_path = train_data["image_path"][i]

        for model_name in model_names:
            # 创建任务ID
            task_id = create_task_id(model_name, i, content)
            all_tasks.append({
                "task_id": task_id,
                "model_name": model_name,
                "content": content,
                "image_path": image_path,
                "target": target,
                "data_index": i
            })

    # 过滤已完成的任筽
    pending_tasks = [task for task in all_tasks if task["task_id"] not in completed_task_ids]

    logger.info(f"总任务数: {len(all_tasks)}, 已完成: {len(completed_task_ids)}, 待处理: {len(pending_tasks)}")

    if len(pending_tasks) == 0:
        logger.info("所有任务已完成，跳过处理")
        # 加载并返回已有结果
        if os.path.exists(result_file):
            return pd.read_csv(result_file)
        return pd.DataFrame()

    # 加载已有结果（如果有）
    if os.path.exists(temp_result_file) and len(completed_task_ids) > 0:
        result_df = pd.read_csv(temp_result_file)
    else:
        result_df = pd.DataFrame(columns=[
            "task_id", "model_name", "input", "image_path", "output",
            "target", "flag", "status", "timestamp", "error"
        ])

    # 创建线程安全的进度条
    pbar = tqdm(total=len(pending_tasks), desc="处理进度")
    completed_count = len(completed_task_ids)
    lock = threading.Lock()

    # 包装函数，固定部分参数
    evaluate_func = partial(
        evaluate_model_wrapper,
        openai_client=openai_client,
        PROMPT=PROMPT,
        model_process=model_process,
        diagnostic_judge=diagnostic_judge,
        encode_image=encode_image
    )

    def task_callback(future, task_detail):
        """任务完成回调函数"""
        nonlocal completed_count, result_df, pbar, completed_task_ids

        try:
            result = future.result()
            task_id = task_detail["task_id"]

            with lock:
                # 标记任务为完成
                completed_task_ids.add(task_id)
                completed_count += 1

                # 添加结果到DataFrame
                new_row = pd.DataFrame([result])
                result_df = pd.concat([result_df, new_row], ignore_index=True)

                # 更新进度条
                pbar.update(1)
                success_count = len(result_df[result_df['status'] == 'success'])
                failed_count = len(result_df[result_df['status'] == 'failed'])
                pbar.set_postfix({
                    "完成": completed_count,
                    "成功率": f"{success_count / completed_count * 100:.1f}%" if completed_count > 0 else "0%"
                })

                # 定期保存临时结果
                if completed_count % checkpoint_interval == 0 or exit_event.is_set():
                    # 保存临时结果
                    try:
                        result_df.to_csv(temp_result_file, index=False, encoding='utf-8')
                        logger.debug(f"临时结果已保存: {temp_result_file} ({completed_count} 个任务)")

                        # 保存进度信息
                        save_progress_info(
                            output_dir,
                            len(all_tasks),
                            len(completed_task_ids),
                            failed_count
                        )
                    except Exception as e:
                        logger.error(f"保存临时结果失败: {str(e)}")

                # 检查是否需要退出
                if should_exit or exit_event.is_set():
                    logger.info("正在优雅退出...")
                    # 保存最终结果
                    result_df.to_csv(result_file, index=False, encoding='utf-8')
                    logger.info(f"结果已保存到: {result_file}")
                    pbar.close()
                    return

        except Exception as e:
            logger.error(f"任务回调处理失败: {str(e)}")
            with lock:
                completed_count += 1
                pbar.update(1)

    # 使用ThreadPoolExecutor并发处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for task_detail in pending_tasks:
            # 检查是否需要退出
            if should_exit or exit_event.is_set():
                logger.info("检测到退出信号，停止提交新任务")
                break

            # 提交任务
            future = executor.submit(
                evaluate_func,
                task_detail["model_name"],
                task_detail["content"],
                task_detail["image_path"],
                task_detail["target"],
                task_detail["task_id"]
            )

            # 添加回调
            future.add_done_callback(
                lambda f, td=task_detail: task_callback(f, td)
            )

            futures.append(future)

            # 控制并发提交速度
            if len(futures) >= max_workers * 2:
                # 等待一些任务完成
                concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                # 移除已完成的任务
                futures = [f for f in futures if not f.done()]

        # 等待所有任务完成
        try:
            if not should_exit and not exit_event.is_set():
                concurrent.futures.wait(futures)
        except KeyboardInterrupt:
            logger.info("收到中断信号，等待任务完成...")
            # 等待正在执行的任务完成
            for future in futures:
                try:
                    future.result(timeout=2)
                except:
                    pass

    pbar.close()

    # 保存最终结果
    try:
        if not result_df.empty:
            result_df.to_csv(result_file, index=False, encoding='utf-8')
            logger.info(f"最终结果已保存到: {result_file}")

            # 删除临时文件
            if os.path.exists(temp_result_file):
                os.remove(temp_result_file)
                logger.debug(f"已删除临时文件: {temp_result_file}")

            # 删除进度文件
            progress_file = os.path.join(output_dir, "progress.json")
            if os.path.exists(progress_file):
                os.remove(progress_file)
                logger.debug(f"已删除进度文件: {progress_file}")

            success_count = len(result_df[result_df['status'] == 'success'])
            failed_count = len(result_df[result_df['status'] == 'failed'])
            logger.info(f"处理完成! 成功: {success_count}, 失败: {failed_count}")

    except Exception as e:
        logger.error(f"保存最终结果失败: {str(e)}")
        # 如果最终保存失败，至少保存临时结果
        if os.path.exists(temp_result_file):
            os.rename(temp_result_file, result_file)
            logger.info(f"已从临时文件恢复结果: {result_file}")

    return result_df


if __name__ == '__main__':
    # 配置参数
    output_dir = "./output/model_eval"
    max_workers = 3  # 根据您的API限制调整并发数
    checkpoint_interval = 5  # 每完成5个任务保存一次检查点

    # 使用并发版本
    result_df = process_batch_parallel(
        openai_client=openai_client,
        train_data=train_data,
        model_names=model_names,
        output_dir=output_dir,
        model_process=model_process,
        diagnostic_judge=diagnostic_judge,
        encode_image=encode_image,
        PROMPT=PROMPT,
        max_workers=max_workers,
        checkpoint_interval=checkpoint_interval
    )

    # 打印统计信息
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"总任务数: {len(result_df)}")
    print(f"成功: {len(result_df[result_df['status'] == 'success'])}")
    print(f"失败: {len(result_df[result_df['status'] == 'failed'])}")

    print(f"结果文件: {os.path.join(output_dir, 'model_results.csv')}")

    # 显示数据框信息
    if not result_df.empty:
        print("\n数据框信息:")
        print(result_df.info())

        # 显示前几行
        print("\n前5行数据:")
        print(result_df.head())
    else:
        print("\n没有处理任何数据")
