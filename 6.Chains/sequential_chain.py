from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt1 = PromptTemplate(
    template="provide me a detailed report on {topic}" ,
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template="provide me 5 most important points from this {report}" ,
    input_variables=['report']
)

parser = StrOutputParser()


chain = prompt1 | model | parser | prompt2 | model | parser 
result = chain.invoke({'topic' : 'black hole'})
print(result)