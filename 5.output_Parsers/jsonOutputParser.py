from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()

parser = JsonOutputParser()

llm = HuggingFaceEndpoint(
    model="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="give me name and age of a character from 3 idiots bollywood movie \n {format_instructions}" ,
    input_variables=[],
    partial_variables={'format_instructions' : parser.get_format_instructions()}
)

chain = prompt | model | parser

result = chain.invoke({})
print(result)