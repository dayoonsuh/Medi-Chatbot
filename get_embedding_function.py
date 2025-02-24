from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
os.environ['OPENAI_API_KEY'] = "YOUR API KEY"
def get_embedding_function():
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    return embeddings
