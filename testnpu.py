import os
import torch
import torch_npu

from transformers import AutoModelForCausalLM, AutoTokenizer

device = "npu:0"

use_jit_compile = os.environ.get("JIT_COMPILE", "0").lower() in ["true", "1"]
torch.npu.set_compile_mode(jit_compile=use_jit_compile)

model_path = '/home/ma-user/work/models/qwen/CodeQwen1___5-7B-Chat'

use_fast_tokenizer = True

revision = 'main'

from_pretrained_kwargs = {'torch_dtype': torch.float32, 'revision': 'main'}

def load_model():  
    tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    use_fast=use_fast_tokenizer,
                    revision=revision,
                    trust_remote_code=True,
                )
    model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    **from_pretrained_kwargs,
                )
    model.to(device)
    print("model load successfully")
    return model, tokenizer
    
def do_infer(model, tokenizer, prompt):
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    print("tokenizer input successfully")
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

if __name__ == '__main__':
    model, tokenizer = load_model()
    infer = do_infer(model, tokenizer, '你是谁?')
    print(infer)