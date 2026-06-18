from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_community.document_loaders import TextLoader, PyPDFLoader , DirectoryLoader
load_dotenv()

loader = DirectoryLoader(
    path="pdfss" ,
    glob ="*.pdf" ,
    loader_cls=PyPDFLoader
)
doc = loader.load()
r=[]
result = None
for d in doc:
    r.append(d.page_content)
    result = " ".join(r)

print(result)
    
