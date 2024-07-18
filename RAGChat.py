import os
import platform
from enum import Enum
import requests
import json

base_url = "http://127.0.0.1:7861"

query_knowledge = []

model_name = "元小猿"
model_company = "吉大正元"
welcome = f"欢迎使用 {model_company} {model_name} 模型，输入内容即可进行对话，clear 清空对话历史，exit 终止程序，soi 信息出处（Sources of information），minfo 模型信息，klist 知识库列表"
system_prompt = f"你是 {model_name}，一个由 {model_company} 打造的人工智能助手，你可以提供多种多样的服务，比如翻译、写代码、闲聊、为您答疑解惑等。"

# history = [{"role": "system", "content": system_prompt}]
history = []

class QueryType(Enum):
    knowledgeList = 0
    modelInfo = 1
    sourceInformation = 2
    chatMessage = 3

def build_prompt(query):
    global history
    messages = {
        "query": query, 
        "knowledge_base_name": "jit", 
        "stream": "True",
        "history": history,
        # "score_threshold": "0.5"
    }
    return messages

def append_history(query, generated_text):
    global history
    user_content = {"role": "user", "content": query}
    assistant_content = {'role': 'assistant', 'content': generated_text}
    history.append(user_content)
    history.append(assistant_content)

def build_headers():
    headers = {
        'Content-Type': 'application/json',
        'accept': 'application/json',
    }
    return headers

def knowledge_list_chat(api="/knowledge_base/list_files?knowledge_base_name=jit"):
    global query_knowledge
    model_url = f"{base_url}{api}"
    response = requests.get(model_url, headers=build_headers())
    if response.status_code == 200:
        print("知识库中包含以下文件：\n")
        for file_index, file_item in enumerate(response.json().get("data")):
            print(f"{file_index + 1}、{file_item}\n")
            query_knowledge = []
    else:
        print('请求失败，状态码：', response.status_code)
        
def model_info_chat(api="/llm_model/list_running_models"):
    global query_knowledge
    model_url = f"{base_url}{api}"
    response = requests.post(model_url, headers=build_headers())
    if response.status_code == 200:
        print("当前运行的大模型为", end='')
        for run_model in response.json().get("data").keys():
            print(f"{run_model} ")
            query_knowledge = []
    else:
        print('请求失败，状态码：', response.status_code)
        
def source_info_chat():
    global query_knowledge
    if len(query_knowledge) > 0:
        print("以上查询的知识来源如下：\n")
        for sourceItem in query_knowledge:
            print(f"{sourceItem}\n")
    else:
        print("以上查询未找到相关知识来源。")
        
def knowledge_search_chat(query, api="/chat/knowledge_base_chat"):
    global query_knowledge
    model_url = f"{base_url}{api}"
    query_data = build_prompt(query)
    with requests.post(model_url, json=query_data, headers=build_headers(), stream=True) as response:
        generated_text = ""
        for line in response.iter_content(None, decode_unicode=True):
            try:
                data = json.loads(line[6:])
                if "answer" in data:
                    print(data["answer"], end="")
                    generated_text += data["answer"]
                if "docs" in data and len(data["docs"]) > 0:
                    query_knowledge = data["docs"]
            except:
                continue
        append_history(query, generated_text)
    
def stream_chat(query, query_type):
    global query_knowledge
    headers = {
        'Content-Type': 'application/json',
        'accept': 'application/json',
    }
    if query_type == QueryType.knowledgeList:
        knowledge_list_chat()
    elif query_type == QueryType.modelInfo:
        model_info_chat()
    elif query_type == QueryType.sourceInformation:
        source_info_chat()
    elif query_type == QueryType.chatMessage:
        knowledge_search_chat(query)
    print("", flush=True)
    
def stream_chat1(query):
    prompt_messages = build_prompt(query)
    print(prompt_messages)

def main():
    os_name = platform.system()
    clear_command = 'cls' if os_name == 'Windows' else 'clear'
    stop_stream = False
    
    print(welcome)
    while True:
        query = input("\n用户：")
        if query.strip() == "exit":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print(welcome)
            continue
        print(f"\n{model_name}：", end='')
        if query.strip() == 'klist':
            stream_chat(query, QueryType.knowledgeList)
            continue
        elif query.strip() == 'minfo':
            stream_chat(query, QueryType.modelInfo)
            continue
        elif query.strip() == 'soi':
            stream_chat(query, QueryType.sourceInformation)
            continue
        stream_chat(query, QueryType.chatMessage)

if __name__ == '__main__':
    main()