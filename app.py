"""
-*- coding: utf-8 -*-

@Author  : houcg
@Time    : 2026/1/12 16:03
"""

from config import *
from utils.llm_util import model_process, encode_image
from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass
import logging
import base64
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DoctorOpinion:
    """医生的意见"""
    model_name: str
    role: str
    opinion: str
    is_final: bool = False
    diagnosis_result: str = ""


class MultiModelConsultation:
    """多模型会诊系统"""

    def __init__(self, model_config: Dict, patient_info: str, image_path: str, max_rounds: int = 10):
        """
        初始化会诊系统

        Args:
            model_config: 模型配置
            patient_info: 患者信息
            image_path: 图片路径
            max_rounds: 最大讨论轮数
        """
        self.model_config = model_config
        self.patient_info = patient_info
        self.image_path = image_path
        self.max_rounds = max_rounds

        # 检查图片是否存在
        if not os.path.exists(image_path):
            logger.warning(f"图片文件不存在: {image_path}")
            self.image_base64 = None
        else:
            # 编码图片为base64
            self.image_base64 = self._encode_image(image_path)

        # 获取主导医生和其他医生
        self.chief_doctor = None
        self.other_doctors = []

        for model_name, config in model_config.items():
            if config.get("main", False):
                self.chief_doctor = model_name
            else:
                self.other_doctors.append(model_name)

        if not self.chief_doctor:
            raise ValueError("未找到主导医生配置")

        self.discussion_history: List[Dict] = []
        self.final_diagnosis = None

    def _encode_image(self, image_path: str) -> Optional[str]:
        """将图片编码为base64字符串"""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

            # 获取图片格式
            if image_path.lower().endswith('.jpg') or image_path.lower().endswith('.jpeg'):
                mime_type = "image/jpeg"
            elif image_path.lower().endswith('.png'):
                mime_type = "image/png"
            else:
                mime_type = "image/jpeg"  # 默认

            return f"data:{mime_type};base64,{encoded_string}"
        except Exception as e:
            logger.error(f"图片编码失败: {e}")
            return None

    def _build_prompt_for_doctor(
            self,
            model_name: str,
            discussion_context: str = None,
            round_number: int = 1
    ):
        """
        为医生构建提示词

        Args:
            model_name: 模型名称
            discussion_context: 讨论上下文
            round_number: 当前轮次

        Returns:
            包含prompt和image的字典
        """
        config = self.model_config[model_name]
        prompt_template = config["prompt"]
        role = config["role"]

        # 构建基础提示词
        full_prompt = f"""
{prompt_template}

### Patient information
{self.patient_info}

### Current round
Round {round_number} Discussion
"""
        if discussion_context:
            full_prompt += f"""
### Discuss history
{discussion_context}
"""

        if model_name == self.chief_doctor:
            full_prompt += """
### Please output in the specified format
Please strictly output your analysis opinions in the following JSON format:：
{
    "is_final": 0或1,
    "analysis_opinion": "Your analytical opinion",
    "diagnosis_result": "Diagnosis result (fill in if is_final=1)"
}
"""
        else:
            full_prompt += """
Please output your analysis opinions
Please directly output your professional analysis opinions. No specific format is required.
"""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self.image_base64
                        },
                    },
                    {"type": "text", "text": full_prompt},
                ],
            },
        ]

        return messages

    def _parse_chief_doctor_response(self, response: str) -> Dict:
        """解析主导医生的响应"""
        try:
            # 尝试从响应中提取JSON
            response = response.strip()

            # 如果响应包含代码块，提取JSON部分
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].strip()
            else:
                json_str = response

            data = json.loads(json_str)

            # 验证必需字段
            if "is_final" not in data or "analysis_opinion" not in data:
                raise ValueError("响应缺少必需字段")

            return {
                "is_final": bool(data["is_final"]),
                "analysis_opinion": data["analysis_opinion"],
                "diagnosis_result": data.get("diagnosis_result", "")
            }

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"解析主导医生响应失败: {e}")
            # 尝试提取可能的JSON
            if "{" in response and "}" in response:
                try:
                    # 尝试找到JSON部分
                    start_idx = response.find('{')
                    end_idx = response.rfind('}') + 1
                    json_str = response[start_idx:end_idx]
                    data = json.loads(json_str)

                    if "is_final" in data and "analysis_opinion" in data:
                        return {
                            "is_final": bool(data["is_final"]),
                            "analysis_opinion": data["analysis_opinion"],
                            "diagnosis_result": data.get("diagnosis_result", "")
                        }
                except:
                    pass

            # 返回默认值
            return {
                "is_final": False,
                "analysis_opinion": response,
                "diagnosis_result": ""
            }

    def _format_discussion_context(self) -> str:
        """格式化讨论上下文"""
        if not self.discussion_history:
            return "This is the first round of discussion and there is no historical record yet"

        context_parts = []
        for i, discussion in enumerate(self.discussion_history, 1):
            context_parts.append(f"第{i}轮讨论:")

            for opinion in discussion["opinions"]:
                role = self.model_config[opinion.model_name]["role"]
                # 截断过长的意见
                truncated_opinion = opinion.opinion
                if len(truncated_opinion) > 500:
                    truncated_opinion = truncated_opinion[:497] + "..."
                context_parts.append(f"  {role}（{opinion.model_name}）: {truncated_opinion}")

                if opinion.is_final:
                    context_parts.append(f"  [Final diagnostic recommendation]: {opinion.diagnosis_result}")

            context_parts.append("")

        return "\n".join(context_parts)

    def _call_model(self, model_name: str, chat_messages) -> str:
        """
        调用模型

        Args:
            model_name: 模型名称
            prompt_data: 包含prompt和图片的数据

        Returns:
            模型响应
        """
        try:
            # 根据模型类型处理图片
            response = model_process(openai_client, chat_messages, model_name)
            return response
        except Exception as e:
            logger.error(f"调用模型 {model_name} 失败: {e}")
            return f"[模型 {model_name} 调用失败]: {str(e)}"

    def run_consultation(self) -> Dict[str, Any]:
        """
        执行会诊讨论

        Returns:
            会诊结果
        """
        logger.info("开始多模型会诊...")

        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"=== 第{round_num}轮讨论开始 ===")

            # 获取讨论上下文
            discussion_context = self._format_discussion_context()

            # 步骤1: 其他医生先发表意见
            other_opinions = []
            for doctor in self.other_doctors:
                logger.info(f"{self.model_config[doctor]['role']} ({doctor}) 发表意见...")

                # 为其他医生构建提示词
                doctor_prompt_data = self._build_prompt_for_doctor(
                    doctor,
                    discussion_context,
                    round_num
                )
                doctor_response = self._call_model(doctor, doctor_prompt_data)

                other_opinions.append({
                    "model_name": doctor,
                    "role": self.model_config[doctor]["role"],
                    "opinion": doctor_response
                })

                logger.info(f"{self.model_config[doctor]['role']} 意见收集完成")

            # 步骤2: 汇总其他医生的意见
            other_opinions_summary = "\n### 其他医生意见汇总:\n"
            for i, opinion in enumerate(other_opinions, 1):
                other_opinions_summary += f"{i}. {opinion['role']} ({opinion['model_name']}): {opinion['opinion']}\n"

            # 步骤3: 主导医生最后发言，综合所有意见
            logger.info(f"主导医生 {self.chief_doctor} 综合意见并总结...")

            # 为主导医生构建包含其他医生意见的提示词
            chief_context = discussion_context
            if chief_context:
                chief_context += other_opinions_summary
            else:
                chief_context = other_opinions_summary

            # 添加总结要求
            chief_context += """
    ### Please take into account the opinions of the above doctors and provide the final analysis
    1. Please analyze the opinions of other doctors and point out which ones are reasonable and which ones need further consideration
    2. Based on all the information and discussions, provide your final analysis
    3. If you think a definite diagnosis can be made, please set is_final to 1 and give the diagnosis result
    4. If you think more information or discussion is needed, please set is_final to 0 and indicate what other aspects of information are required
    """

            chief_prompt_data = self._build_prompt_for_doctor(
                self.chief_doctor,
                chief_context,
                round_num
            )
            chief_response = self._call_model(self.chief_doctor, chief_prompt_data)
            chief_opinion = self._parse_chief_doctor_response(chief_response)

            # 创建DoctorOpinion对象
            other_opinion_objs = []
            for opinion in other_opinions:
                other_opinion_objs.append(DoctorOpinion(
                    model_name=opinion["model_name"],
                    role=opinion["role"],
                    opinion=opinion["opinion"]
                ))

            chief_opinion_obj = DoctorOpinion(
                model_name=self.chief_doctor,
                role=self.model_config[self.chief_doctor]["role"],
                opinion=chief_opinion["analysis_opinion"],
                is_final=chief_opinion["is_final"],
                diagnosis_result=chief_opinion["diagnosis_result"]
            )

            # 保存本轮讨论
            round_opinions = other_opinion_objs + [chief_opinion_obj]
            self.discussion_history.append({
                "round": round_num,
                "opinions": round_opinions
            })

            # 检查是否结束
            if chief_opinion["is_final"]:
                logger.info(f"主导医生在第{round_num}轮做出最终诊断: {chief_opinion['diagnosis_result']}")
                self.final_diagnosis = {
                    "round": round_num,
                    "diagnosis": chief_opinion["diagnosis_result"],
                    "analysis": chief_opinion["analysis_opinion"],
                    "chief_doctor": self.chief_doctor,
                    "other_opinions_summary": other_opinions_summary
                }

                # 记录其他医生的意见
                for opinion in other_opinions:
                    self.final_diagnosis[f"{opinion['role']}_opinion"] = opinion["opinion"][:500] + (
                        "..." if len(opinion["opinion"]) > 500 else "")

                break

            # 检查是否达到最大轮数
            if round_num >= self.max_rounds:
                logger.warning(f"已达到最大讨论轮数 ({self.max_rounds})，但未达成最终诊断")
                # 在最后一轮，基于所有意见做最终判断
                self.final_diagnosis = {
                    "round": round_num,
                    "diagnosis": "未达成明确诊断",
                    "analysis": chief_opinion["analysis_opinion"],
                    "chief_doctor": self.chief_doctor,
                    "other_opinions_summary": other_opinions_summary,
                    "status": "max_rounds_reached"
                }

                # 记录其他医生的意见
                for opinion in other_opinions:
                    self.final_diagnosis[f"{opinion['role']}_opinion"] = opinion["opinion"][:500] + (
                        "..." if len(opinion["opinion"]) > 500 else "")

                break

            logger.info(f"=== 第{round_num}轮讨论结束，未达成最终诊断，继续下一轮 ===\n")

        # 如果没有达成最终诊断
        if not self.final_diagnosis and self.discussion_history:
            logger.warning(f"经过{len(self.discussion_history)}轮讨论未达成最终诊断")
            # 使用最后一轮主导医生的意见作为最终诊断
            last_round = self.discussion_history[-1]
            chief_opinion = last_round["opinions"][-1]  # 最后一个意见是主导医生的

            # 收集最后一轮其他医生的意见
            other_opinions_summary = "\n### 最后一轮其他医生意见:\n"
            for opinion in last_round["opinions"][:-1]:  # 排除主导医生
                other_opinions_summary += f"{opinion.role}: {opinion.opinion[:300]}...\n"

            self.final_diagnosis = {
                "round": last_round["round"],
                "diagnosis": "未达成明确诊断",
                "analysis": chief_opinion.opinion,
                "chief_doctor": self.chief_doctor,
                "other_opinions_summary": other_opinions_summary,
                "status": "inconclusive"
            }

        return self._generate_final_report()

    def _generate_final_report(self) -> Dict[str, Any]:
        """生成最终报告"""
        report = {
            "final_diagnosis": self.final_diagnosis,
            "discussion_summary": self._generate_summary(),
            "total_rounds": len(self.discussion_history),
            "all_rounds": [],
            "patient_info": self.patient_info[:500] + ("..." if len(self.patient_info) > 500 else ""),
            "image_available": self.image_base64 is not None
        }

        for discussion in self.discussion_history:
            round_info = {
                "round": discussion["round"],
                "opinions": []
            }

            for opinion in discussion["opinions"]:
                round_info["opinions"].append({
                    "doctor": opinion.model_name,
                    "role": self.model_config[opinion.model_name]["role"],
                    "opinion": opinion.opinion[:300] + ("..." if len(opinion.opinion) > 300 else ""),
                    "opinion_full": opinion.opinion,
                    "is_final": opinion.is_final,
                    "diagnosis_result": opinion.diagnosis_result
                })

            report["all_rounds"].append(round_info)

        return report

    def _generate_summary(self) -> str:
        """生成讨论总结"""
        summary_parts = [f"会诊总结（共{len(self.discussion_history)}轮讨论）："]

        for discussion in self.discussion_history:
            summary_parts.append(f"\n第{discussion['round']}轮讨论：")
            for opinion in discussion["opinions"]:
                role = self.model_config[opinion.model_name]["role"]
                summary_parts.append(f"  - {role}: {opinion.opinion[:150]}...")

        if self.final_diagnosis:
            summary_parts.append(f"\n最终诊断：{self.final_diagnosis['diagnosis']}")
            if "status" in self.final_diagnosis and self.final_diagnosis["status"] == "inconclusive":
                summary_parts.append("注意：此为未达成一致意见的初步判断")

        return "\n".join(summary_parts)


# 使用示例
def run_consultation(patient_info: str, image_path: str) -> Dict[str, Any]:
    """
    运行会诊的主函数

    Args:
        patient_info: 患者信息字符串
        image_path: 图片路径

    Returns:
        会诊结果
    """
    # 导入模型配置
    from config import model_config

    # 创建会诊系统
    consultation_system = MultiModelConsultation(
        model_config=model_config,
        patient_info=patient_info,
        image_path=image_path,
        max_rounds=10
    )

    # 运行会诊
    result = consultation_system.run_consultation()

    # 输出结果
    if result["final_diagnosis"]:
        logger.info(f"\n=== 会诊结果 ===")
        logger.info(f"轮次: 第{result['final_diagnosis']['round']}轮")
        logger.info(f"诊断: {result['final_diagnosis']['diagnosis']}")
        logger.info(f"分析: {result['final_diagnosis']['analysis'][:300]}...")

    return result


# 工具函数
def validate_model_config(config: Dict) -> bool:
    """验证模型配置"""
    required_fields = ["role", "main", "prompt"]

    for model_name, model_config in config.items():
        for field in required_fields:
            if field not in model_config:
                raise ValueError(f"模型 {model_name} 缺少必需的字段: {field}")

    # 检查是否有且仅有一个主导医生
    chief_doctors = [name for name, cfg in config.items() if cfg.get("main", False)]
    if len(chief_doctors) != 1:
        raise ValueError(f"应该有且仅有一个主导医生，找到 {len(chief_doctors)} 个")

    return True


# 主函数
def main():
    """主函数"""
    # 患者信息
    patient_info = """
A woman aged 60 years presented, in 2023, with progressive deviation of her occlusion, although she experienced no pain. A panoramic x-ray revealed birateral osteolysis of the mandible. Contrast-enhanced CT demonstrated progressive mandibular bone resorption extending from the mandibular body to the condylar regions, prompting referral to the departments of rheumatology, haematology, and oral surgery for further evaluation. She had no other symptoms, and clinically there were no signs of arthritis (ie, no joint swelling or tenderness). However, hand radiographs obtained as part of systemic screening prompted by the mandibular findings subsequently revealed an unusual pattern of osteolysis. Laboratory tests showed a normal C-reactive protein concentration (0·10 mg/dL) and matrix metalloproteinase-3 (20·8 ng/mL), although tartrate-resistant acid phosphatase type 5b was markedly elevated (851 U/L), suggesting substantial osteoclastic activity. Antinuclear antibody was positive at 1:320 (speckled and cytoplasmic pattern) and anti-Sjögren's syndrome-related antigen A antibody was also positive, raising the suspicion of underlying Sjögren's disease. Immunofixation electrophoresis revealed no monoclonal protein, ruling out multiple myeloma. Other skeletal disorders (eg, enchondromatosis), metabolic disorders (eg, hyperparathyroidism), and rheumatic diseases (eg, sarcoidosis, psoriatic arthritis, and rheumatoid arthritis) were also excluded. Biopsy of the mandible showed massive osteolysis and abnormal lymphatic proliferation. In combination with the progressive osteolysis demonstrated radiographically, the patient was diagnosed as Gorham–Stout disease.
"""

    # 图片路径
    image_path = "./data/images/lancet_images/What is the key radiographic feature.jpg"

    try:
        # 验证配置
        validate_model_config(model_config)

        # 检查图片文件是否存在
        if not os.path.exists(image_path):
            logger.warning(f"警告: 图片文件不存在: {image_path}")
            # 使用默认图片或继续运行
            image_path = None

        # 同步会诊
        result = run_consultation(patient_info, image_path)

        # 输出完整结果
        print("\n=== 完整会诊报告 ===")
        print(f"患者信息: {result.get('patient_info', 'N/A')}")
        print(f"图片可用: {result.get('image_available', False)}")
        print(f"讨论轮次: {result.get('total_rounds', 0)}")

        if result.get("final_diagnosis"):
            print(f"\n最终诊断: {result['final_diagnosis'].get('diagnosis', 'N/A')}")
            print(f"诊断轮次: 第{result['final_diagnosis'].get('round', 0)}轮")
            print(f"分析意见: {result['final_diagnosis'].get('analysis', 'N/A')}")

        print(f"\n讨论总结:\n{result.get('discussion_summary', 'N/A')}")

        return result

    except Exception as e:
        logger.error(f"会诊执行失败: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    result = main()

    # 如果需要保存结果到文件
    if result and "error" not in result:
        output_file = "consultation_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"会诊结果已保存到: {output_file}")
