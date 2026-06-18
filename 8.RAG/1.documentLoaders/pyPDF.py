from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_community.document_loaders import TextLoader, PyPDFLoader
load_dotenv()

# llm = HuggingFaceEndpoint(
#     model="deepseek-ai/DeepSeek-V4-Flash",
#     task="text-generation"
# )
# parser = StrOutputParser()

# model = ChatHuggingFace(llm=llm)

loader = PyPDFLoader(
    file_path="F:\LangChain Models\8.RAG\elect.pdf"
    )
document = loader.load()
page = document[0].page_content
print(page)


# if I wanted to load whole pdf
# r=[]
# result = None
# for d in doc:
#     r.append(d.page_content)
#     result = " ".join(r)

# print(result)