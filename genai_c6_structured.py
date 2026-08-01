from pyclbr import Class

import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field
from uuid import uuid4
import os
import base64
import mimetypes

load_dotenv()
from typing import List, Optional
class Recipe(BaseModel):
    name: str = Field(description="Name of the recipe")
    description: str = Field(description = "Brief description of the recipe")
    prep_time: str = Field(description="Estimated preparation time of the recipe")

class Response(BaseModel):
    """A single recipe"""
    ingredients: List[str] = Field(description="List of main ingredients")
    recipes: List[Recipe] = Field(description="List of 3 recipe suggestions")

image_path = "images/receipe.avif"
mime_type, _ = mimetypes.guess_type(image_path)

with open(image_path, 'rb') as file:
    raw_binary = file.read()
    base64_bytes = base64.b64encode(raw_binary)
    image_base64 = base64_bytes.decode("utf-8")

# Initialize Gemini Flash 2.5
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.7,
)
system_prompt = """
You are a helpful chef. Identify the main ingredients. Suggest 3 recipes based those ingredients.
"""



if __name__ == "__main__":
    with SqliteSaver.from_conn_string("receipe_teller.db") as checkpointsaver:

        config = {"configurable": {"thread_id": "1"}}
        agent = create_agent(
            model=llm,
            system_prompt=system_prompt,
            response_format=Response,
        )
        message = [{"role": "system", "content": system_prompt},
                   {"role": "user", "content": [{"type": "image", "base64": image_base64, "mime_type": mime_type}]}]

        response = agent.invoke({"messages":message}, config=config)
        result=response["structured_response"]
        print("\nIngredients:")
        for ingredient in result.ingredients:
            print(f"- {ingredient}")

        print("\nRecipes:")
        for index, recipe in enumerate(result.recipes, start=1):
            print(f"\n{index}. {recipe.name}")
            print(f"Description: {recipe.description}")
            print(f"Prep time: {recipe.prep_time}")