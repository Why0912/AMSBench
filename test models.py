
import base64
import json
import requests
import re
import os  # 用于检查文件是否存在
import time  # 用于延迟重试
from PIL import Image
import io
import numpy as np

# 读取本地图片并编码为Base64字符串
def encode_image(image_path):
    """
    读取本地图片并编码为Base64字符串。
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def encode_image_from_pil(img, format="PNG"):
    """
    从 PIL 图像对象直接编码为 Base64 字符串，不保存到磁盘。
    """
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return encoded_string



# 清理模型回答中的特殊符号和空格，只保留选项字母
def clean_model_answer(model_answer):
    """
    清理模型回答中的特殊符号和空格，只保留选项字母。
    """
    # 打印原始输入以便调试
    print("原始输入（可能包含不可见字符）：", repr(model_answer))
    
    # 按行分割回答，确保处理多行文本
    lines = model_answer.strip().split('\n')
    print("按行分割后的内容：", lines)
    
    # 从最后一行开始向前查找，直到找到包含字母的内容
    for line in reversed(lines):
        print("当前检查的行：", repr(line))
        # 使用正则表达式提取字母（A-Z 或 a-z）
        match = re.search(r'[A-Za-z]', line)
        if match:
            extracted_letter = match.group(0).upper()
            print("提取到的字母：", extracted_letter)
            return extracted_letter  # 返回找到的字母并转换为大写
    print("未找到任何字母，返回 'none'")
    return 'none'  # 如果没有找到字母，返回 'none'


# 计算MSE
def calculate_mse(true_values, predicted_values):
    valid_true = []
    valid_pred = []
    for t, p in zip(true_values, predicted_values):
        if p is not None:  # 只考虑有效的模型回答
            valid_true.append(t)
            valid_pred.append(p)
    if valid_true:
        return np.mean((np.array(valid_true) - np.array(valid_pred)) ** 2)
    else:
        return None  # 如果没有有效的数据，返回None

# 从字符串中提取数字
def extract_number_from_string(text):
    match = re.search(r'\d+', text)
    if match:
        return int(match.group(0))  # 返回提取的第一个数字
    return None  # 如果没有找到数字，返回None

# 提取模型回答的最后一行，并从中获取数字
def extract_components_count_from_answer(answer_text):
    lines = answer_text.strip().split('\n')
    last_line = lines[-1]  # 获取最后一行
    print("最后一行文本：", last_line)
    
    # 将文本按空格分割成单词
    words = last_line.split()

    for word in reversed(words):
        model_answer = extract_number_from_string(word)
        if model_answer is not None:
            return model_answer  # 找到数字并返回
    return None  # 如果没有找到数字，返回None

# 封装API请求，加入重试机制
def send_request_with_retry(payload, headers, max_retries=5, delay=3):
    attempt = 0
    while attempt < max_retries:
        try:
            # 发送请求
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()  # 如果响应状态码不是 200，将抛出异常
            return response.json()  # 返回响应内容

        except (requests.exceptions.RequestException, KeyError, Exception) as e:
            print(f"请求失败: {e}，正在重试... 第 {attempt + 1} 次")
            attempt += 1
            if attempt < max_retries:
                time.sleep(delay)  # 等待几秒钟后重试
            else:
                print("达到最大重试次数，跳过此请求")
                return None  # 如果超过最大重试次数，返回 None

# 处理数据并生成结果
def process_data(json_data, api_url, api_key):
    correct_count = 0
    total_count = 0
    
    level_correct = {"Easy": 0, "Medium": 0, "Hard": 0}
    level_total = {"Easy": 0, "Medium": 0, "Hard": 0}

    # 进度文件路径，用于记录已处理的索引
    progress_file_path = r"progressed.txt"
    
    # 结果文件路径
    result_file_path = r"results.txt"

    # 读取已处理的索引（如果进度文件存在）
    processed_indices = set()
    if os.path.exists(progress_file_path):
        with open(progress_file_path, "r", encoding="utf-8") as progress_file:
            for line in progress_file:
                try:
                    index = int(line.strip())
                    processed_indices.add(index)
                except ValueError:
                    continue  # 忽略无效的行

    # 以追加模式打开结果文件（避免覆盖之前的记录）
    with open(result_file_path, "a", encoding="utf-8") as result_file:
        with open(progress_file_path, "a", encoding="utf-8") as progress_file:
            for index, item in enumerate(json_data):  # 使用 enumerate 获取索引
                # 如果该索引已经处理过，跳过
                if index in processed_indices:
                    print(f"已跳过处理的索引: {index}")
                    continue

                image_path = item["image_path"]
                question = item["question"]
                options = item["options"]
                groundtruth = item["groundtruth"]
                level = item["level"]

                # 直接读取图片并编码为 Base64
                try:
                    base64_image = encode_image(image_path)
                except Exception as e:
                    print(f"图片处理失败：{image_path}, 错误：{e}, 跳过该图片")
                    progress_file.write(f"{index}\n")  # 即使失败也记录索引，避免下次重复处理
                    result_file.write(f"{image_path} (Index: {index}) Error in image processing\n")
                    progress_file.flush()
                    result_file.flush()
                    continue

            

                # 创建请求数据payload
                payload = {
                    "model": "model_name",  # 替换为实际模型名称
                    "messages": [
                        {"role": "system", "content": "Please answer according to the images, questions, and options provided."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f'''{question}{options}.The final response format as“**<correct option letter>**”'''},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    },
                                },
                            ],
                        },
                    ],
                    "temperature": 0.1,
                    "user": "",
                }

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": f"",
                }

                # 发送API请求（带重试机制）
                response = send_request_with_retry(payload, headers)

                # 获取模型的回答
                if response and response.get("choices"):
                    model_answer = response.get("choices", [])[0].get("message", {}).get("content", "").strip()
                    print(model_answer)  # 打印模型回答

                    # 确保 model_answer 被成功拆分
                    split_answer = model_answer.split()
                    if split_answer:
                        model_answer = split_answer[-1]  # 提取最后的选项字母
                    else:
                        model_answer = 'none'  # 如果拆分后是空列表，设置为 'none'
                    
                    model_answer = clean_model_answer(model_answer)
                    print(model_answer)

                    # 检查是否正确
                    is_correct = model_answer == groundtruth
                    if is_correct:
                        correct_count += 1
                        level_correct[level] += 1
                    level_total[level] += 1

                    # 实时写入结果
                    result_file.write(f"{image_path} (Index: {index}) {groundtruth} {model_answer} {level} {'Correct' if is_correct else 'Incorrect'}\n")
                else:
                    result_file.write(f"{image_path} (Index: {index}) Error in response\n")

                # 记录已处理的索引
                progress_file.write(f"{index}\n")
                progress_file.flush()  # 确保写入磁盘，防止程序中断时数据丢失
                result_file.flush()  # 确保结果也写入磁盘

                total_count += 1

        # 计算正确率（仅统计已处理的项）
        overall_accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
        level_accuracies = {
            level: (level_correct[level] / level_total[level]) * 100 if level_total[level] > 0 else 0
            for level in level_correct
        }

        # 输出正确率
        result_file.write("\nOverall Accuracy: {:.2f}%\n".format(overall_accuracy))
        for level, accuracy in level_accuracies.items():
            result_file.write(f"{level} Accuracy: {accuracy:.2f}%\n")

# 读取JSON数据
json_file_path = r"C:\Users\why18\Desktop\EDA\AMSnet2.0\AMSBench\final_json\connection900.json"  # 替换为你的JSON文件路径
with open(json_file_path, "r", encoding="utf-8") as json_file:
    data = json.load(json_file)

# 设置API的基础URL和API Key

api_url = ""
api_key = ""  # 替换为你的API Key

# 调用处理函数
process_data(data, api_url, api_key)
