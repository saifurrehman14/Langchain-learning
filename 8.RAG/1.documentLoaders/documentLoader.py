from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_community.document_loaders import TextLoader
load_dotenv()

loader = TextLoader(file_path="F:\LangChain Models\8.RAG\crickPoem.txt")

doc = loader.load()

llm = HuggingFaceEndpoint(
    model="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation"
)
parser = StrOutputParser()

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="generate summary of this poem: {poem}",
    input_variables=['poem']
)




chain = prompt | model | parser
result = chain.invoke({'poem' : doc[0].page_content})
print(result)