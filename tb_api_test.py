#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import time
import re
import requests

# ======================= 用户需要配置的部分 =======================

API_URL                 = ""
API_KEY                 = ""

PROMPT_TEMPLATE_PATH    = "prompt.md"
TESTBENCH_EXAMPLE_PATH  = "unit_cap.txt"
TB_CSV_PATH             = "TB_problem.csv"
OUTPUT_DIR              = "LDO"

SINGLE_NETLIST_FILE     = "LDO.txt"
TEST_SINGLE             = True
SINGLE_CIRCUIT          = "LDO"
ONLY_METRICS            = ["Dropout Voltage", "DC gain", "Phase maigin", "PSR", "offset"]


MODEL_NAME            = " "  
# ==============================================================

def load_tb_csv(path: str):
    """读取 CSV，返回 [(circuit, [metric,...]), ...]"""
    tasks = []
    with open(path, newline="", encoding="utf-8") as cf:
        reader = csv.reader(cf)
        next(reader)
        for row in reader:
            circ = row[0].strip()
            metrics = [m.strip() for m in row[1].split(",") if m.strip()]
            tasks.append((circ, metrics))
    return tasks


def call_api(messages, temperature=0.7):

    payload = json.dumps({
        "model": MODEL_NAME,
        "temperature": temperature,
        "messages": messages,
        "user": ""
    })
    headers = {
        'Accept': 'application/json',
        'Authorization': API_KEY,
        'User-Agent': f'API/1.0.0 ({API_KEY})',
        'Content-Type': 'application/json'
    }
    resp = requests.request("POST", API_URL, headers=headers, data=payload)
    resp.raise_for_status()
    data = resp.json()
    return data['choices'][0]['message']['content']


def extract_and_save(response_text, circuit, metric, output_dir):
    """
    按三反引号分隔提取前 5 段代码，保存为 .cir 文件
    """
    blocks = re.findall(r"```(?:plaintext)?\n([\s\S]*?)\n```", response_text)
    print(f"[DEBUG] {circuit}_{metric}: found {len(blocks)} code block(s)")
    if not blocks:
        print(f"[!] 没有找到任何代码块，原始响应：\n{response_text}")
        return

    for i, netlist in enumerate(blocks[:5], start=1):
        fname = f"{circuit}_{metric}_answer{i}.cir"
        path = os.path.join(output_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(netlist.strip() + "\n")
        print(f"[+] 已保存 {path}")


def main():
    # 读取并拼接 prompt 模板与示例
    with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        base_prompt = f.read().strip()
    with open(TESTBENCH_EXAMPLE_PATH, "r", encoding="utf-8") as f:
        example_tb = f.read().strip()
    base_prompt = base_prompt + "\n\n" + example_tb + "\n\n"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tasks = load_tb_csv(TB_CSV_PATH)

    if TEST_SINGLE:
        tasks = [t for t in tasks if t[0] == SINGLE_CIRCUIT]
        if not tasks:
            print(f"[!] CSV 中未找到电路 '{SINGLE_CIRCUIT}'")
            return

    MAX_RETRIES = 3
    for circuit, metrics in tasks:
        print(f"\n=== Processing {circuit} ===")
        if TEST_SINGLE:
            netlist_file = SINGLE_NETLIST_FILE
        else:
            netlist_file = os.path.join(os.path.dirname(TB_CSV_PATH), f"{circuit}.txt")

        if not os.path.isfile(netlist_file):
            print(f"[!] 未找到网表 {netlist_file}")
            continue

        for metric in metrics:
            if ONLY_METRICS and metric not in ONLY_METRICS:
                continue

            prompt_text = (
                base_prompt
                .replace("certain circuit", circuit)
                .replace("certain performance", metric)
            )
            netlist_input = (
                f"The file path of the circuit netlist for which the testbench is to be generated: {netlist_file}"
            )

            # 将 prompt_text 和 netlist_input 分开为两条 user 消息
            messages = [
                # {"role": "system", "content": "You are an analog integrated circuits expert."},
                {"role": "user", "content": "You are an analog integrated circuits expert.\n\n" + prompt_text},
                {"role": "user", "content": netlist_input}
            ]

            success = False
            for attempt in range(1, MAX_RETRIES + 1):
                print(f"[*] Calling model {MODEL_NAME} for {circuit}-{metric} (try {attempt}/{MAX_RETRIES})…")
                try:
                    resp = call_api(messages, temperature=0.7)
                except Exception as e:
                    print(f"[!] API 调用失败: {e}")
                    if attempt < MAX_RETRIES:
                        time.sleep(1.0)
                    continue

                blocks = re.findall(r"```(?:plaintext)?\n([\s\S]*?)\n```", resp)
                print(f"[DEBUG] {circuit}_{metric}: found {len(blocks)} code block(s)")

                if len(blocks) < 5:
                    print(f"[!] 只收到了 {len(blocks)} 段代码（需要 5 段），准备重试…")
                    if attempt < MAX_RETRIES:
                        time.sleep(1.0)
                    continue

                subdir = os.path.join(OUTPUT_DIR, f"{circuit}_{metric}")
                os.makedirs(subdir, exist_ok=True)
                extract_and_save(resp, circuit, metric, subdir)
                success = True
                break

            if not success:
                print(f"[!] {circuit}_{metric} 没拿到 5 段，兜底保存已有 {len(blocks)} 段")
                subdir = os.path.join(OUTPUT_DIR, f"{circuit}_{metric}")
                os.makedirs(subdir, exist_ok=True)
                extract_and_save(resp, circuit, metric, subdir)

if __name__ == "__main__":
    main()
