from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
# llm = HuggingFaceEndpoint(
#     model="deepseek-ai/DeepSeek-V4-Flash" ,
#     task="text-generation"
# )

# model = ChatHuggingFace(llm = llm)

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

prompt1 = PromptTemplate(
    template="write me a detailed report on {topic}" ,
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="give me 5 most important words from this text {text}" ,
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser 

result = chain.invoke({'topic': 'black-hole'})

print(result)