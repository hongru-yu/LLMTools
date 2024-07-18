import os

import torch
import torch_npu

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

import json

torch_device = "npu:0"

use_jit_compile = os.environ.get("JIT_COMPILE", "0").lower() in ["true", "1"]
torch.npu.set_compile_mode(jit_compile=use_jit_compile)

def load_model():   
    # model_dir = "/home/ma-user/work/models/JITCodeYXY-7B-Chat_dpo_1"
    model_dir = "/home/ma-user/work/models/JITCodeYXY-7B-Chat_lora_sft"

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
    
    return model, tokenizer

def do_infer(model, tokenizer, prompt):
    messages = [
        # {"role": "system", "content": "你是 元小猿，一个由 吉大正元 打造的人工智能助手，你可以提供多种多样的服务，比如翻译、写代码、闲聊、为您答疑解惑等。"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    model_inputs = tokenizer([text], return_tensors="pt").to(torch_device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        do_sample=False,
        temperature=0.95,
        top_p=0.7, 
        top_k=50, 
        num_beams=1, 
        max_new_tokens=1024, 
        repetition_penalty=1.0, 
        length_penalty=1.0
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
def data_conversion(input_file, output_file, model, tokenizer):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    converted_data = []
    for item in data:
        infer = do_infer(model, tokenizer, item["instruction"])
        print("当前问题 : ", item["instruction"])
        print("正确答案 : ", item["output"])
        print("推理结果 : ", infer)
        print("==============================================")  
        if infer == item["output"]:
            infer = "";
            print("与正确答案的比对结果 : 一致")
        else:
            print("与正确答案的比对结果 : 不一致")
        print("==============================================") 
        converted_item = {
            "conversations": [
                {
                    "from": "human",
                    "value": item["instruction"]
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": item["output"]
            },
            "rejected": {  
                "from": "gpt",
                "value": infer
            }
        }
        converted_data.append(converted_item)
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(converted_data, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    model, tokenizer = load_model()
    input_file = "/home/ma-user/work/LLaMA-Factory/data/identity.json"
    output_file = "/home/ma-user/work/LLaMA-Factory/data/identity_dpo.json"
    data_conversion(input_file, output_file, model, tokenizer)
    input_file = "/home/ma-user/work/LLaMA-Factory/data/pkitool.json"
    output_file = "/home/ma-user/work/LLaMA-Factory/data/pkitool_dpo.json"
    data_conversion(input_file, output_file, model, tokenizer)
