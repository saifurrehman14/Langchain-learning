from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda ,RunnablePassthrough

load_dotenv()

llm = HuggingFaceEndpoint(
    model="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation"
)
parser = StrOutputParser()

model1 = ChatHuggingFace(llm=llm)

model2 = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")


prompt1 = PromptTemplate(
    template="generate a joke on this topic: {topic}" , 
    input_variables=['topic']
)

chain1 = RunnableSequence(prompt1 , model1 , parser)

def count_words(text):
    textt = text.split()
    r = len(textt)
    return r


word_counter_runnable = RunnableLambda(count_words)

passTheText = RunnablePassthrough()

result = RunnableParallel({
    'pass' : RunnableSequence(chain1 , passTheText)
    ,
    'count' : RunnableSequence(chain1 , word_counter_runnable)
})

print(result.invoke({'topic' : 'football'}))