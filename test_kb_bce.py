from BCEmbedding.tools.langchain import BCERerank

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.retrievers import ContextualCompressionRetriever

import sentence_transformers

# export LD_PRELOAD=/home/ma-user/miniconda3/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0


# init embedding model
embedding_model_name = '/home/ma-user/work/models/maidalun/bce-embedding-base_v1'
embedding_model_kwargs = {'device': 'npu:0'}
embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': False}

embed_model = HuggingFaceEmbeddings(
  model_name=embedding_model_name,
  model_kwargs=embedding_model_kwargs,
  encode_kwargs=embedding_encode_kwargs
)

reranker_args = {'model': '/home/ma-user/work/models/maidalun/bce-reranker-base_v1', 'top_n': 5, 'device': 'npu:1'}
reranker = BCERerank(**reranker_args)

# init documents
documents = PyPDFLoader("/home/ma-user/work/knowledge_base/商用密码应用安全性评估FAQ第一版.pdf").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# example 1. retrieval with embedding and reranker
retriever = FAISS.from_documents(texts, embed_model, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT).as_retriever(search_type="similarity", search_kwargs={"score_threshold": 0.3, "k": 10})

compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=retriever
)

response = compression_retriever.get_relevant_documents("什么是密评?")
print("response : ", response)
print("========================================================================")

response = compression_retriever.get_relevant_documents("密评的流程是什么?")
print("response : ", response)
print("========================================================================")

response = compression_retriever.get_relevant_documents("密评都需要保护什么?")
print("response : ", response)
print("========================================================================")