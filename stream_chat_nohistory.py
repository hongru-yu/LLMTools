import os
import platform
import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, InfNanRemoveLogitsProcessor, LogitsProcessorList
from transformers.generation.utils import GenerationConfig
from threading import Thread
import json

torch_device = "npu:0"

use_jit_compile = os.environ.get("JIT_COMPILE", "0").lower() in ["true", "1"]
torch.npu.set_compile_mode(jit_compile=use_jit_compile)

# model_dir = "/home/ma-user/work/models/JITCodeYXY-7B-Chat_dpo_1"
# model_dir = "/home/ma-user/work/models/baicai003/llama-3-8b-Instruct-chinese_v2"
# model_dir = "/home/ma-user/work/models/qwen/CodeQwen1___5-7B-Chat"
# model_dir = "/home/ma-user/work/models/JITCodeYXY-7B-Chat_lora_sft"
model_dir = "/home/ma-user/work/models/JITCodeYXW-7B-Chat_lora_sft"

history = []

model_name = "元小猿"
model_company = "吉大正元"
welcome = f"欢迎使用 {model_company} {model_name} 模型，输入内容即可进行对话，clear 清空对话历史，exit 终止程序"
system_prompt = f"你是 {model_name}，一个由 {model_company} 打造的人工智能助手，你可以提供多种多样的服务，比如翻译、写代码、闲聊、为您答疑解惑等。"

def load_model():   
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, 
        trust_remote_code=True,
        use_fast_tokenizer = True,
        split_special_tokens = False,
        cache_dir = None, 
        revision = 'main', 
        token = None
    )

    print("tokenizer load successfully")

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True, 
        cache_dir=None, 
        revision='main', 
        token=None, 
        low_cpu_mem_usage=True, 
        torch_dtype=torch.float16, 
        device_map='auto', 
        offload_folder='offload'
    ).to(torch_device)

    model.requires_grad_(False)
    model.eval()

    model.generation_config = GenerationConfig.from_pretrained(model_dir)

    print("model load successfully")
    print("\n----------------------------------------------------------------------------\n")
    
    return model, tokenizer

def build_prompt(query, hasHistory, hasSystem):
    messages = [{"role": "user", "content": query}]
    if hasHistory:
        messages = history + messages
    if hasSystem:
        messages = [{"role": "system", "content": system_prompt}] + messages
    return messages

def append_history(query, generated_text):
    user_content = {"role": "user", "content": query}
    assistant_content = {'role': 'assistant', 'content': generated_text}
    history.append(user_content)
    history.append(assistant_content)

def stream_chat(model, tokenizer, query):
    prompt_messages = build_prompt(query, False, True)
    
    text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt", return_token_type_ids=False).to(torch_device)

    generating_args = dict(
        do_sample=False,
        length_penalty=1.0,
        eos_token_id=[tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=1024,
    )
    
    
    logits_processor = LogitsProcessorList()
    logits_processor.append(InfNanRemoveLogitsProcessor())
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # generation_kwargs = dict(inputs=model_inputs.input_ids, streamer=streamer, generation_config=GenerationConfig(**generating_args), logits_processor=logits_processor)
    generation_kwargs = dict(model_inputs, streamer=streamer, generation_config=GenerationConfig(**generating_args), logits_processor=logits_processor)
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
    thread.start()
    
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        print(new_text, end='')
    print("", flush=True)
    append_history(query, generated_text)

def main():
    os_name = platform.system()
    clear_command = 'cls' if os_name == 'Windows' else 'clear'
    stop_stream = False
    
    model, tokenizer = load_model()
    
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
        stream_chat(model, tokenizer, query)

if __name__ == '__main__':
    main()