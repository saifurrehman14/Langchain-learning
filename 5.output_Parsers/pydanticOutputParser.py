from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import Field, BaseModel
from langchain_core.prompts import PromptTemplate


load_dotenv()

llm = HuggingFaceEndpoint(
    model = "deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class employee(BaseModel):
    name :  str = Field(description="name of the employee")
    age : int = Field(gt=18, description="age of the employee")
    city : str = Field(description="city of the employee")

parser = PydanticOutputParser(pydantic_object=employee)


prompt = PromptTemplate(
    template= "extract the name, age and city of the employee from the following information: {information} \n {format_instruction}" ,
    input_variables= ['information'] , 
    partial_variables= {'format_instruction' : parser.get_format_instructions()}

)

chain = prompt | model | parser
result = chain.invoke({'information': "In a quiet office filled with the soft hum of computers, a new employee had just joined the team. His name was Zavian Cole, a thoughtful and observant individual who preferred listening over speaking. At 29 years old, he had already built a reputation for solving complex problems with calm precision. He had recently moved from Eldridge Haven, a city known for its misty mornings and creative energy, bringing with him a fresh perspective that quickly made him stand out among his colleagues."})
print(result)
