from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda ,RunnablePassthrough , RunnableBranch

load_dotenv()

llm = HuggingFaceEndpoint(
    model="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation"
)
parser = StrOutputParser()

model1 = ChatHuggingFace(llm=llm)

model2 = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")


prompt1 = PromptTemplate(
    template="generate me a report on this topic: {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="generate me a summary of this text: {text}",
    input_variables=['text']
)

chain1 = RunnableSequence(prompt1 , model1 , parser)


passText = RunnablePassthrough()

chain2 = RunnableBranch(
    (lambda x: len(x.split()) <500 , passText),
    (lambda x: len(x.split()) >500 , prompt2 | model1| parser),
    RunnableLambda(lambda x: "invalid input")

)

chain3 = RunnableSequence(chain1, chain2)
print(chain3.invoke({'topic'  : 'black-hole'}))