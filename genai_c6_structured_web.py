import base64
import os
import uuid
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "structured-recipe-dev-secret")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing from .env")


class Recipe(BaseModel):
    name: str = Field(description="Name of the recipe")
    description: str = Field(description="Brief description of the recipe")
    prep_time: str = Field(description="Estimated preparation time of the recipe")
    quantity: str = Field(description="Quantity of the ingredients", default=None)
    dish_palace: str = Field(description="Dish palace where the recipe is from", default=None)


class Response(BaseModel):
    ingredients: list[str] = Field(description="List of main ingredients")
    recipes: list[Recipe] = Field(description="List of 3 recipe suggestions")


SYSTEM_PROMPT = """
You are a helpful chef.

Tasks:
1. Identify the main visible ingredients from the uploaded image.
2. Suggest exactly 3 recipe ideas based on those ingredients.
3. Keep ingredient names concise.
4. Keep recipe descriptions practical and short.
5. Never invent details that are clearly not supported by the image.
6. Dish came from which palace
"""


app = Flask(__name__)
app.secret_key = SESSION_SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False,
)


llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.4,
)

agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    response_format=Response,
)


def initialize_session() -> None:
    if "thread_id" not in session:
        session["thread_id"] = str(uuid.uuid4())

    if "recipe_history" not in session:
        session["recipe_history"] = []


def is_async_request() -> bool:
    return request.headers.get("X-Requested-With") == "XMLHttpRequest"


def encode_uploaded_image(uploaded_file: Any) -> tuple[str, str]:
    if uploaded_file is None or not uploaded_file.filename:
        raise ValueError("Please choose an image to analyze.")

    mime_type = uploaded_file.mimetype or ""

    if not mime_type.startswith("image/"):
        raise ValueError("Only image uploads are supported.")

    image_bytes = uploaded_file.read()

    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

    return base64.b64encode(image_bytes).decode("utf-8"), mime_type


def invoke_recipe_agent(image_base64: str, mime_type: str, thread_id: str) -> Response:
    message = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "base64": image_base64,
                    "mime_type": mime_type,
                }
            ],
        },
    ]

    response = agent.invoke(
        {"messages": message},
        {"configurable": {"thread_id": thread_id}},
    )

    structured_response = response.get("structured_response")

    if isinstance(structured_response, Response):
        return structured_response

    if isinstance(structured_response, dict):
        return Response.model_validate(structured_response)

    raise ValueError("The model did not return a structured recipe response.")


def build_history_entry(filename: str, result: Response) -> dict[str, Any]:
    return {
        "filename": filename,
        "ingredients": result.ingredients,
        "recipes": [
            {
                "name": recipe.name,
                "description": recipe.description,
                "prep_time": recipe.prep_time,
                "quantity":recipe.quantity,
                "dish_palace":recipe.dish_palace

            }
            for recipe in result.recipes
        ],
    }


@app.route("/")
def landing_page() -> str:
    return render_template("recipe_home.html")


@app.route("/recipe-vision")
def home() -> str:
    initialize_session()
    return render_template(
        "structured_recipe.html",
        history=session.get("recipe_history", []),
    )


@app.route("/analyze", methods=["POST"])
def analyze_image():
    initialize_session()

    uploaded_file = request.files.get("image")

    try:
        image_base64, mime_type = encode_uploaded_image(uploaded_file)
        result = invoke_recipe_agent(
            image_base64=image_base64,
            mime_type=mime_type,
            thread_id=session["thread_id"],
        )
        history_entry = build_history_entry(uploaded_file.filename, result)

        history = list(session.get("recipe_history", []))
        history.insert(0, history_entry)
        session["recipe_history"] = history[:6]
        session.modified = True

    except (ValueError, ValidationError) as exc:
        if is_async_request():
            return jsonify({"success": False, "error": str(exc)}), 400

        return redirect(url_for("home"))

    except Exception as exc:
        print(f"Structured recipe error: {exc}")

        if is_async_request():
            return jsonify(
                {
                    "success": False,
                    "error": "Sorry, I could not analyze that image right now.",
                }
            ), 500

        return redirect(url_for("home"))

    if is_async_request():
        return jsonify(
            {
                "success": True,
                "result": history_entry,
            }
        )

    return redirect(url_for("home"))


@app.route("/clear")
def clear_results():
    session.clear()
    return redirect(url_for("home"))


@app.route("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "structured-recipe-web",
    }


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True,
    )
