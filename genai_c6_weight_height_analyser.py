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
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "weight-height-dev-secret")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing from .env")


class PersonResponse(BaseModel):
    age: int = Field(description="Estimated age of the person in years")
    height: float = Field(description="Estimated height of the person in meters")
    weight: float = Field(description="Estimated weight of the person in kilograms")
    gender: str = Field(description="Estimated gender presentation of the person")
    diet_suggestion: str = Field(description="Short healthy diet suggestion for the person")


SYSTEM_PROMPT = """
You are a helpful health and fitness assistant.
Analyze the uploaded person image and return:
1. Estimated height in meters.
2. Estimated weight in kilograms.
3. Estimated age in years.
4. Estimated gender presentation.
5. A short healthy diet suggestion based on the estimated body profile.

Only return the structured response fields.
"""


app = Flask(__name__)
app.secret_key = SESSION_SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.3,
)

agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    response_format=PersonResponse,
)


def initialize_session() -> None:
    if "thread_id" not in session:
        session["thread_id"] = str(uuid.uuid4())

    if "analysis_history" not in session:
        session["analysis_history"] = []


def is_async_request() -> bool:
    return request.headers.get("X-Requested-With") == "XMLHttpRequest"


def encode_uploaded_image(uploaded_file: Any, label: str) -> tuple[str, str]:
    if uploaded_file is None or not uploaded_file.filename:
        raise ValueError(f"Please choose {label}.")

    mime_type = uploaded_file.mimetype or ""
    if not mime_type.startswith("image/"):
        raise ValueError(f"{label.capitalize()} must be an image.")

    image_bytes = uploaded_file.read()
    if not image_bytes:
        raise ValueError(f"{label.capitalize()} is empty.")

    return base64.b64encode(image_bytes).decode("utf-8"), mime_type


def invoke_agent(image_base64: str, mime_type: str, thread_id: str) -> PersonResponse:
    message = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Estimate this person's body profile from the image.",
                },
                {
                    "type": "image",
                    "base64": image_base64,
                    "mime_type": mime_type,
                },
            ],
        },
    ]

    response = agent.invoke(
        {"messages": message},
        {"configurable": {"thread_id": thread_id}},
    )

    structured_response = response.get("structured_response")

    if isinstance(structured_response, PersonResponse):
        return structured_response

    if isinstance(structured_response, dict):
        return PersonResponse.model_validate(structured_response)

    raise ValueError("The model did not return a structured person analysis.")


def build_person_result(filename: str, result: PersonResponse) -> dict[str, Any]:
    return {
        "filename": filename,
        "age": result.age,
        "height": result.height,
        "weight": result.weight,
        "gender": result.gender,
        "diet_suggestion": result.diet_suggestion,
    }


def build_history_entry(first_person: dict[str, Any], second_person: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": f"{first_person['filename']} and {second_person['filename']}",
        "people": [first_person, second_person],
    }


@app.route("/")
def landing_page() -> str:
    initialize_session()
    return render_template(
        "weight_calculator.html",
        history=session.get("analysis_history", []),
    )


@app.route("/analyze", methods=["POST"])
def analyze_image():
    initialize_session()

    uploaded_file_1 = request.files.get("image_1")
    uploaded_file_2 = request.files.get("image_2")

    try:
        image_1_base64, image_1_mime = encode_uploaded_image(uploaded_file_1, "the first image")
        image_2_base64, image_2_mime = encode_uploaded_image(uploaded_file_2, "the second image")

        first_result = build_person_result(
            uploaded_file_1.filename,
            invoke_agent(image_1_base64, image_1_mime, session["thread_id"]),
        )
        second_result = build_person_result(
            uploaded_file_2.filename,
            invoke_agent(image_2_base64, image_2_mime, session["thread_id"]),
        )

        history_entry = build_history_entry(first_result, second_result)
        history = list(session.get("analysis_history", []))
        history.insert(0, history_entry)
        session["analysis_history"] = history[:6]
        session.modified = True

    except (ValueError, ValidationError) as exc:
        if is_async_request():
            return jsonify({"success": False, "error": str(exc)}), 400
        return redirect(url_for("landing_page"))

    except Exception as exc:
        print(f"Weight/height analysis error: {exc}")
        if is_async_request():
            return jsonify(
                {
                    "success": False,
                    "error": "Sorry, I could not analyze those images right now.",
                }
            ), 500
        return redirect(url_for("landing_page"))

    if is_async_request():
        return jsonify(
            {
                "success": True,
                "result": history_entry,
            }
        )

    return redirect(url_for("landing_page"))


@app.route("/clear")
def clear_results():
    session.clear()
    return redirect(url_for("landing_page"))


@app.route("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "weight-height-analyser",
    }


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True,
    )
