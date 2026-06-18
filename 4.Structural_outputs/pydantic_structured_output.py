from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel , Field
from typing import Literal
load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

class employeeDetails(BaseModel):
    performance : Literal["strongly Pos", "Strongly Neg", "Strongly Neurtal"]  =  Field(description="His overall performance this year, based on the number of projects he completed, behaviour with other employees.")
    age : int = Field(default=None, description="employee's age")
    salary : float = Field(default = None, description="salary of employee")
    projects : list[str] = Field(default=None, description="Name pf Projects, he workes on this year")

structured_output = model.with_structured_output(employeeDetails)
result = structured_output.invoke("Employee Review File | Name: Hira Malik | Age: mid-20s (HR lists 26) | Projects: Data Cleaning Pipeline ✔, Recommendation Engine ✔ (but later scalability issues), A/B Testing Framework ✔ | Complications: The recommendation system worked initially but failed under high load (post-deployment issue). | Behavior: Very cooperative and positive attitude, frequently seeks help before attempting solutions independently, active in meetings but decision-making confidence is low | Peer Comments: Hira is great to work with, but needs to improve independent thinking. | Salary: ~140k PKR/month | Final Thought: Good team member, but still developing core problem-solving ability.")
print(result.performance)
print(result.age)
print(result.salary)
print(result.projects)