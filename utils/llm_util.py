"""
-*- coding: utf-8 -*-

@Author  : houcg
@Time    : 2026/1/8 10:23
"""
import base64

from loguru import logger


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def model_process(model_client, messages, model_name="gpt-5-chat-latest"):
    try:
        response = model_client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        # logger.error(f"model_name:{model_name} messages:{messages}")
        logger.error(e)
    return None


def diagnostic_judge(model_client, target, output, model_name="gpt-5-mini"):
    if target == output:
        return 1
    for _ in range(3):
        try:
            prompt = f"""
###Task
Determine whether the following diagnostic pairs are consistent. Output only 0 (consistent) or 1 (inconsistent).

### Example
    Example 1
        Standard diagnosis: Acute appendicitis
        To be evaluated: Appendicitis, acute
    Output: 0

    Example 2
        Standard diagnosis: Type 2 diabetes
        Under assessment: Diabetes
    Output: 0

    Example 3
        Standard diagnosis: Community-acquired pneumonia
        Pending assessment: Tuberculosis
    Output: 1
    
    Example 4
        Standard diagnosis: Actinomycosis
        Pending assessment: 1、Actinomycosis
    Output: 1

Now please judge:
Standard diagnosis: {target}
To be evaluated: {output}
            """
            messages = [{"role": "user", "content": prompt}]

            response = model_client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            return int(response.choices[0].message.content)
        except Exception as e:
            logger.error(e)
    return 0
