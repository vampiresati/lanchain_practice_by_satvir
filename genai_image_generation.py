import os
from dotenv import load_dotenv
from google import genai
from PIL import Image
from io import BytesIO
from urllib.parse import quote

load_dotenv()

from langchain.tools import tool
import requests
# client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
#
# response = client.models.generate_content(
#     model="gemini-3.1-flash-lite-image",
#     contents="A cartoon character doing meditation in a forest"
# )
#
# for part in response.candidates[0].content.parts:
#     if getattr(part, "inline_data", None):
#         img = Image.open(BytesIO(part.inline_data.data))
#         img.save("tiger.png")
#         print("Saved tiger.png")


@tool
def generate_image(prompt: str):
    """Generate an image."""
    url = f"https://image.pollinations.ai/prompt/{quote(prompt)}"
    img = requests.get(url, timeout=30).content

    with open("image.png", "wb") as f:
        f.write(img)

    return "Saved image.png"

if __name__ == "__main__":
    print(
        generate_image.invoke(
            {"prompt": "A clothed man doing meditation in a forest"}
        )
    )
