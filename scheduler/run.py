import os
import requests
from openai import OpenAI

THUNDER_AGENT_URL = "http://172.16.33.142:9000" 
MODEL_NAME = "Qwen3-32B" 

def run_thunder_task(program_id: str, prompt: str):
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"{THUNDER_AGENT_URL}/v1"
    )

    print(f"--- 正在启动程序: {program_id} ---")

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "你是一个高效的 AI 助手。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        stream=True,
        extra_body={
            "program_id": program_id,
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
    )
    print(f"\n[Output]:\n")
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            print(text, end="", flush=True)
    print("\n")
    profile_res = requests.get(f"{THUNDER_AGENT_URL}/profiles/{program_id}")
    if profile_res.status_code == 200:
        print(f"[性能统计]: {profile_res.json()}")

    release_res = requests.post(
        f"{THUNDER_AGENT_URL}/programs/release",
        json={"program_id": program_id}
    )
    if release_res.status_code == 200:
        print(f"--- 程序 {program_id} 已成功释放 ---")

if __name__ == "__main__":
    pid = "task_001"
    prompt = "请解释一下什么是大模型的 KV Cache 优化。"
    
    run_thunder_task(pid, prompt)