import os
import zipfile


def compress_folder_to_zip(folder_path, output_zip_path):

    if not output_zip_path.endswith('.zip'):
        output_zip_path += '.zip'

    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            relative_path = os.path.relpath(root, folder_path)
            for file in files:
                zipf.write(os.path.join(root, file), os.path.join(relative_path, file))


if __name__ == '__main__':
    # folder_to_compress = 'Langchain-Chatchat-npu'  # 替换为你想要压缩的文件夹路径
    # output_zip_file = 'Langchain-Chatchat.zip'  # 指定输出的zip文件名
    folder_to_compress = 'LLaMA-Factory'  # 替换为你想要压缩的文件夹路径
    output_zip_file = 'LLaMA-Factory.zip'  # 指定输出的zip文件名
    compress_folder_to_zip(folder_to_compress, output_zip_file)
    print(f"Folder compressed successfully to {output_zip_file}")