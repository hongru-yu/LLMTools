from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

llm = ChatOpenAI(
    streaming=True,
    verbose=True,
    # callbacks=[callback],
    openai_api_key="none",
    openai_api_base="http://127.0.0.1:8000/v1",
    model_name="CodeQwen1.5-7B-Chat"
)



# 提示词
template = """
我很想去{location}旅行，我应该在哪里做什么？
"""
prompt = PromptTemplate(
    input_variables=["location"],
    template=template,

)
# 说白了就是在提示词的基础上，把输入的话进行格式化方法输入，前后添加了一些固定词
final_prompt = prompt.format(location='安徽合肥')

print(f"最终提升次：{final_prompt}")
output = llm([HumanMessage(content=final_prompt)])
print(f"LLM输出结果：{output}")

