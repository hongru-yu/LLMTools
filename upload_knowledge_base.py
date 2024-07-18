import os
import requests
from tqdm import tqdm

def upload_txt_files(folder_path, upload_url, knowledge_base_name):
    # 初始化一个空列表，用于存储找到的txt文件的路径
    txt_files = []
    # 初始化一个变量，用于记录找到的txt文件总数
    total_files = 0
    
    # 使用os.walk遍历指定文件夹及其子文件夹
    for root, dirs, files in os.walk(folder_path):
        # 遍历当前文件夹中的所有文件
        for file in files:
            # 如果文件是txt文件，则将其路径添加到txt_files列表中
            if file.endswith(".json"):
                txt_files.append(os.path.join(root, file))
    # 更新找到的txt文件总数
    total_files = len(txt_files)
    # 打印找到的txt文件总数
    print("Total txt files found:", total_files)

    # 初始化一个变量，用于记录已上传的文件数量
    uploaded_count = 0
    
    with tqdm(total=total_files, desc="Uploading") as pbar:
        # 遍历所有找到的txt文件
        for txt_file in txt_files:
            # 准备上传的文件，包括文件名、文件对象和文件类型
            # files = {'files': (os.path.basename(txt_file), open(txt_file, 'rb'), 'application/msword')}
            files = {'files': (os.path.basename(txt_file), open(txt_file, 'rb'), 'application/json')}
            # 准备上传的数据，包括一些设置参数和知识库名称
            data = {
                'to_vector_store': 'true',
                'override': 'false',
                'not_refresh_vs_cache': 'false',
                'chunk_size': '500',
                'chunk_overlap': '50',
                'zh_title_enhance': 'true',
                'knowledge_base_name': knowledge_base_name
            }
            # 发送POST请求，上传文件
            response = requests.post(upload_url, files=files, data=data)
            # 更新已上传的文件数量
            uploaded_count += 1
            # 更新进度条
            pbar.update(1)
            # 更新进度条的后缀，显示已上传的文件数量
            pbar.set_postfix({"Uploaded": uploaded_count})
            # 打印服务器的响应
            print("Response:", response.text)

if __name__ == "__main__":
    # 设置需要上传的txt文件所在的文件夹路径
    folder_path = "/home/ma-user/work/knowledge_base/"
    # 设置LangChainChatChat的API上传地址
    upload_url = 'http://127.0.0.1:7861/knowledge_base/upload_docs'
    # 设置要上传到的知识库的名称
    knowledge_base_name = "samples"  
    # 调用函数，开始上传txt文件
    upload_txt_files(folder_path, upload_url, knowledge_base_name)

