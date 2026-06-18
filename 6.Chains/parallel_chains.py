from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

model1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

llm = HuggingFaceEndpoint(
    model="deepseek-ai/DeepSeek-V4-Flash" ,
    task = "text-generation"
)
model2 = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="generate a short notes from this content : {text}" , 
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="generate a 5 question answers quiz from the following content: {text}" , 
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="merge the notes and quiz in a single file,  notes ->{notes}, quiz -> {quiz}" ,
    input_variables=['notes' , 'quiz']
)


chainOne = RunnableParallel(
    {
        'notes' : prompt1 | model1 | parser 
    ,
        'quiz' : prompt2 | model2 | parser
    }
)

chainTwo = prompt3 | model2 | parser

finalChain = chainOne | chainTwo

text = "Supervised learning is a fundamental concept in machine learning where a model learns from labeled data. In this approach, each input is paired with a correct output, which helps the model understand patterns. It works similarly to a student learning under the guidance of a teacher. During training, the model makes predictions and compares them with the actual answers. The difference between predicted and actual values is called error. The model uses this error to improve its performance over time. The main goal is to learn a mapping function from inputs to outputs. Supervised learning is widely used in many real-world applications. It is mainly divided into two types: classification and regression. Classification is used when the output is in categories, such as spam or not spam. Regression is used when the output is a continuous value, like predicting house prices. Common algorithms include linear regression, decision trees, and support vector machines. Neural networks are also used in more complex problems. The quality of the model depends heavily on the quality of the data. Large and accurate datasets usually produce better models. Overfitting is a common issue where the model learns too much from training data. To avoid this, techniques like cross-validation are used. Supervised learning plays a key role in artificial intelligence systems. It helps in making predictions and decisions based on data. Overall, it is one of the most widely used machine learning techniques today"

result = finalChain.invoke({'text': text})

print(result)

