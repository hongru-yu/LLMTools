import os

import torch
import torch_npu

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, InfNanRemoveLogitsProcessor, LogitsProcessorList
from transformers.generation.utils import GenerationConfig

from threading import Thread

import json

torch_device = "npu:0"

use_jit_compile = os.environ.get("JIT_COMPILE", "0").lower() in ["true", "1"]
torch.npu.set_compile_mode(jit_compile=use_jit_compile)

def load_model():   
    model_dir = "/home/ma-user/work/LLaMA-Factory/models/JITCodeYXY-7B-Chat_dpo_1"

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
    
    generation_kwargs = dict(inputs=model_inputs.input_ids, streamer=streamer, generation_config=GenerationConfig(**generating_args), logits_processor=logits_processor)
    
    print("generation_kwargs : ", generation_kwargs)
    thread = Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
    thread.start()
    
    generated_text = ""
    for new_text in streamer:
        print(new_text, end='')
    print("", flush=True)

if __name__ == '__main__':
    model, tokenizer = load_model()
    prompt = "你是谁"
    do_infer(model, tokenizer, prompt)
    prompt = "你是干啥的啊"
    do_infer(model, tokenizer, prompt)