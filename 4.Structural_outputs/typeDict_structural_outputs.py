from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Optional, Annotated , Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

class review(TypedDict):
    summary : Annotated[str, "give me a short summary of the whole review"]
    sentiment : Annotated[Literal["Pos", "Neg", "Neu"],"Based on review tell me wether the overall sentement is positive or negative or neutral"]
    name : Annotated[Optional[str], "name is the person who gave review(if available)"]


structured_output = model.with_structured_output(review)
result = structured_output.invoke("AOA I am saif have a neutral experience using with dominos pizza")
print(result)