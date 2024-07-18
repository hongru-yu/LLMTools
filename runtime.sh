#!/bin/bash

pip install torch==2.2.0

pip install torch-npu==2.2.0

pip install deepspeed==0.13.2

pip install modelscope

pip install decorator

pip install jq

source /usr/local/Ascend/ascend-toolkit/set_env.sh

export USE_MODELSCOPE_HUB=1

cd Langchain-Chatchat-npu

pip install -r requirements.txt

