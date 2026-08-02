import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from uuid import uuid4
import os
from pydantic import BaseModel, Field, ValidationError
load_dotenv()
import base64

class ImageResponse(BaseModel):
    describe_image: str = Field(description="Describe the image")
    area: str = Field(description="Area of the image")
    color: str = Field(description="Color of the image")
    text: str = Field(description="Text in the image")
    object: str = Field(description="Object in the image")
    emotion: str = Field(description="Emotion in the image")
    language: str = Field(description="Language in the image")
    time: str = Field(description="Time in the image")
    weather: str = Field(description="Weather in the image")
    location: str = Field(description="Location in the image")
with open("images/img.png", "rb") as f:
    binary_image=f.read()
    base64_image=base64.b64encode(binary_image)
    base64_image=base64_image.decode("utf-8")
    print(base64_image)
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite",temperature=0.7)


system_prompt = """You are a helpful image descriptor.Describe what you seen in image. """
user_query = "Describe the image"
agent = create_agent(model=llm,system_prompt=system_prompt,response_format=ImageResponse)
response = agent.invoke({"messages": [{"role": "user", "content": user_query}]})
print(response['structured_response'])
describe_image=response['structured_response'].describe_image
area=response['structured_response'].area
color=response['structured_response'].color
text=response['structured_response'].text