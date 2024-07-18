import requests
import json

def get_from_openapi(question):

    url = "http://localhost:8000/v1/chat/completions"

    payload = {
      "model": "/home/ma-user/work/LLaMA-Factory/saves/JITCodeYXY-7B-Chat/dpo/round1",
      "messages": [
        {
          "role": "user",
          "content": question,
          "tool_calls": [
            {
              "id": "call_default",
              "type": "function",
              "function": {
                "name": "string",
                "arguments": "string"
              }
            }
          ]
        }
      ],
      "tools": [
        {
          "type": "function",
          "function": {
            "name": "string",
            "description": "string",
            "parameters": {}
          }
        }
      ],
      "stream": "False",
      "do_sample": "False",
      "temperature": 0.7
    }

    json_payload = json.dumps(payload)

    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.post(url, data=json_payload, headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        return response_data['choices'][0]['message']['content']
    else:
        return response.status_code

def data_conversion(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    converted_data = []
    for item in data:
        infer = get_from_openapi(item["instruction"])
        print("instruction: ", item["instruction"])
        print("output: ", item["output"])
        print("infer: ", infer)
        print("==============================================")
        if infer == item["output"]:
            infer = "";
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
    input_file = "/home/ma-user/work/LLaMA-Factory/data/identity.json"
    output_file = "/home/ma-user/work/LLaMA-Factory/data/identity_dpo.json"
    data_conversion(input_file, output_file)
    input_file = "/home/ma-user/work/LLaMA-Factory/data/pkitool.json"
    output_file = "/home/ma-user/work/LLaMA-Factory/data/pkitool_dpo.json"
    data_conversion(input_file, output_file)