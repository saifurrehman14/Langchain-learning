from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
load_dotenv()

llm = HuggingFaceEndpoint(
    model="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation"
)
parser = StrOutputParser()

model1 = ChatHuggingFace(llm=llm)

model2 = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

prompt1 = PromptTemplate(
    template="generate me an explaination on this topic: {topic}" , 
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template="give me 5 most important points out of this explaiantion {topic}",
    input_variables=['topic']
)

res1 = RunnableParallel({
    'explaination' : RunnableSequence(prompt1, model1 , parser)
    ,
    'points' : RunnableSequence(prompt2 , model2 , parser)
})
print(res1.invoke({'topic' : 'black-hole'}))

prompt4 = PromptTemplate(
    template="merge these 2 texts in a single file: {explaination}, {points}",
    input_variables=['explaination', 'points']
)
res2 = RunnableSequence(prompt4, model1 , parser)

result = RunnableSequence(res1 , res2)
