from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_community.document_loaders import TextLoader, PyPDFLoader , DirectoryLoader, WebBaseLoader
load_dotenv()


url = 'https://www.kicsit.edu.pk/'
loader = WebBaseLoader(url)
docs = loader.load()
r= docs[0].page_content


llm = HuggingFaceEndpoint(
    model="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation"
)
parser = StrOutputParser()

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="answer this question: {question}, from the following content: {content}",
    input_variables=['question' , 'content']
)

chain = prompt | model | parser
result = chain.invoke({'question': 'where is the university located?', 'content' : r})
print(result)