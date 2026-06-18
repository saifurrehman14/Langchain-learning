from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel ,RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser , PydanticOutputParser
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv


load_dotenv()

# model1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

llm = HuggingFaceEndpoint(
    model = "deepseek-ai/DeepSeek-V4-Flash" , 
    task = "text-generation"
)
model1 = ChatHuggingFace(llm = llm)


class review(BaseModel):
    feedback: Literal["Positive" , "Negative"] = Field(description="Based on the user return the sentiment as Positive or negative. Just the output word")

parser = PydanticOutputParser(pydantic_object=review)
parser2 = StrOutputParser()

prompt1 = PromptTemplate(
    template="Based on the user feedback classify whether the semtiment is Positive or negative: {feedback} , {format_instructions}" ,
    input_variables=['feedback'] ,
    partial_variables={'format_instructions' : parser.get_format_instructions()}
)


prompt2 = PromptTemplate(
    template="generate an appropriate response for the user Based on this positive feedback : {response}",
    input_variables=['response']
)
prompt3 = PromptTemplate(
    template="generate an appropriate response for the user Based on this Negative feedback : {response}",
    input_variables=['response']
)
result = RunnableBranch(
    (lambda x:x.feedback=="Positive", prompt2 | model1 | parser2),
    (lambda x:x.feedback=="Negative", prompt3 | model1 | parser2), 
    RunnableLambda(lambda x: "invalid input")

)

f="The mobile is not very good"
chain1 = prompt1 | model1 | parser
chain2 = chain1 | result
result = chain2.invoke({'feedback' : f})
print(result)