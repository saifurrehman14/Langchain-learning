from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
load_dotenv()

llm = HuggingFaceEndpoint(
    model="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation"
)
parser = StrOutputParser()

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="generate me a joke on this topic: {topic}",
    # partial_variables={'format_instructions' : parser.get_format_instructions } ,
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template="explain me this joke: {joke}",
    input_variables=['joke']
)

r = RunnableSequence(prompt, model, parser)
r2 = RunnableSequence(prompt2, model, parser)
r3 = RunnableSequence(r, r2)
result = r3.invoke({'topic' : 'bugatti'})
print(result)