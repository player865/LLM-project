"""
-*- coding: utf-8 -*-

@Author  : houcg
@Time    : 2026/1/8 10:05
"""
import os.path

import pandas
from sklearn.model_selection import train_test_split

# nejm_data = pandas.read_excel("./data/nejm.xlsx")
# lancet_data = pandas.read_excel("./data/lancet_data.xlsx")
#
# result = {"input": [], "target": [], "image_name": [], "image_path": []}
# print(nejm_data.info())
#
# print(lancet_data.info())
#
# for i in range(len(nejm_data)):
#     if pandas.notna(nejm_data["主讲/病例相关描述"][i]) and pandas.notna(nejm_data["诊断结论"][i]) and pandas.notna(
#             nejm_data["图片名称"][i]):
#
#         intput_info = nejm_data["主讲/病例相关描述"][i]
#         target = nejm_data["诊断结论"][i]
#         image_name = nejm_data["图片名称"][i]
#
#         if os.path.isfile(f"./data/images/nejm_images/{image_name}"):
#             result["input"].append(intput_info)
#             result["target"].append(target)
#             result["image_name"].append(image_name)
#             result["image_path"].append(f"./data/images/nejm_images/{image_name}")
#
# for i in range(len(lancet_data)):
#     if pandas.notna(lancet_data["临床描述"][i]) and pandas.notna(lancet_data["正确答案"][i]) and pandas.notna(
#             lancet_data["图片路径"][i]):
#         intput_info = lancet_data["临床描述"][i]
#         target = lancet_data["正确答案"][i]
#         image_name = lancet_data["图片路径"][i].split("\\")[-1]
#
#         if os.path.isfile(f"./data/images/lancet_images/{image_name}"):
#             result["input"].append(intput_info)
#             result["target"].append(target)
#             result["image_name"].append(image_name)
#             result["image_path"].append(f"./data/images/lancet_images/{image_name}")
# print(len(result["input"]))
#
# df = pandas.DataFrame(result)
#
# train, test = train_test_split(df, test_size=0.3, random_state=42)
#
# train.to_csv("./data/train.csv", index=False, index_label=False)
# test.to_csv("./data/test.csv", index=False, index_label=False)

"""
 最新数据
"""
nejm_data = pandas.read_excel("./data/nejm_image_challenge.xlsx")
lancet_data = pandas.read_excel("./data/lancet_data(total).xlsx")

result = {"input": [], "options": [], "target": [], "image_name": [], "image_path": []}
print(nejm_data.info())

print(lancet_data.info())

for i in range(len(nejm_data)):
    if pandas.notna(nejm_data["主讲/病例相关描述"][i]) and pandas.notna(nejm_data["诊断结论"][i]) and pandas.notna(
            nejm_data["图片名称"][i]):

        intput_info = nejm_data["主讲/病例相关描述"][i]
        options = nejm_data["选项"][i]
        target = nejm_data["诊断结论"][i]
        image_name = nejm_data["图片名称"][i]

        if os.path.isfile(f"./data/images/nejm_images/{image_name}"):
            result["input"].append(intput_info)
            result["options"].append(options)
            result["target"].append(target)
            result["image_name"].append(image_name)
            result["image_path"].append(f"./data/images/nejm_images/{image_name}")

nejm_len = len(result["input"])
print("nejm_len:", nejm_len)

for i in range(len(lancet_data)):
    if pandas.notna(lancet_data["临床描述"][i]) and pandas.notna(lancet_data["正确答案"][i]) and pandas.notna(
            lancet_data["图片路径"][i]):
        intput_info = lancet_data["临床描述"][i]
        target = lancet_data["正确答案"][i]
        options = lancet_data["所有选项"][i]
        image_name = lancet_data["图片路径"][i].split("\\")[-1]

        if os.path.isfile(f"./data/images/lancet_images/{image_name}"):
            result["input"].append(intput_info)
            result["options"].append(options)
            result["target"].append(target)
            result["image_name"].append(image_name)
            result["image_path"].append(f"./data/images/lancet_images/{image_name}")
print("lancet_len:", len(result["input"]) - nejm_len)

df = pandas.DataFrame(result)

train, test = train_test_split(df, test_size=0.3, random_state=42)

train.to_csv("./data/train_with_options.csv", index=False, index_label=False)
test.to_csv("./data/test_with_options.csv", index=False, index_label=False)