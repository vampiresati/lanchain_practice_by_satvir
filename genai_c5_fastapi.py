import os
import sqlite3
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from starlette.middleware.sessions import SessionMiddleware

load_dotenv()

SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not SESSION_SECRET_KEY:
    raise RuntimeError("SESSION_SECRET_KEY is missing from .env")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing from .env")


# Stores the browser coordinates for the current request.
# The get_location tool reads these values when called by the agent.
current_coordinates: ContextVar[dict[str, float] | None] = ContextVar(
    "current_coordinates",
    default=None,
)


def get_weather(city: str) -> dict[str, Any]:
    """
    Get live weather for a city using Open-Meteo.

    Open-Meteo does not require an API key.
    Returns Celsius and Fahrenheit so the agent can select the correct unit.
    """
    try:
        # Step 1: Convert the city name into latitude and longitude.
        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": city,
                "count": 1,
                "language": "en",
                "format": "json",
            },
            timeout=10,
        )
        geo_response.raise_for_status()
        geo_data = geo_response.json()

        locations = geo_data.get("results", [])

        if not locations:
            return {
                "success": False,
                "error": f"City '{city}' was not found.",
            }

        location = locations[0]
        latitude = location["latitude"]
        longitude = location["longitude"]

        # Step 2: Retrieve the current weather.
        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "apparent_temperature,"
                    "wind_speed_10m,"
                    "weather_code"
                ),
                "timezone": "auto",
            },
            timeout=10,
        )
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        current = weather_data.get("current")

        if not current:
            return {
                "success": False,
                "error": f"Current weather is unavailable for '{city}'.",
            }

        temperature_celsius = current.get("temperature_2m")

        if temperature_celsius is None:
            return {
                "success": False,
                "error": f"Temperature data is unavailable for '{city}'.",
            }

        temperature_fahrenheit = round(
            (temperature_celsius * 9 / 5) + 32,
            1,
        )

        return {
            "success": True,
            "city": location.get("name", city),
            "country": location.get("country", ""),
            "country_code": location.get("country_code", ""),
            "latitude": latitude,
            "longitude": longitude,
            "temperature_celsius": temperature_celsius,
            "temperature_fahrenheit": temperature_fahrenheit,
            "feels_like_celsius": current.get("apparent_temperature"),
            "humidity": current.get("relative_humidity_2m"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "weather_code": current.get("weather_code"),
        }

    except requests.Timeout:
        return {
            "success": False,
            "error": "The weather service timed out.",
        }

    except requests.RequestException as exc:
        return {
            "success": False,
            "error": f"Could not retrieve weather: {exc}",
        }

    except (KeyError, TypeError, ValueError) as exc:
        return {
            "success": False,
            "error": f"Unexpected weather response: {exc}",
        }


def reverse_geocode(latitude: float, longitude: float) -> dict[str, Any]:
    """
    Convert browser latitude and longitude into a city and country.
    """
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={
                "lat": latitude,
                "lon": longitude,
                "format": "json",
                "addressdetails": 1,
            },
            headers={
                "User-Agent": "WeatherAssistant/1.0"
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        address = data.get("address", {})

        city = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("municipality")
            or address.get("county")
        )

        country = address.get("country", "")
        country_code = address.get("country_code", "").upper()

        if not city:
            return {
                "success": False,
                "error": "A city could not be determined from the coordinates.",
            }

        return {
            "success": True,
            "city": city,
            "country": country,
            "country_code": country_code,
            "location": f"{city}, {country}" if country else city,
        }

    except requests.Timeout:
        return {
            "success": False,
            "error": "The location service timed out.",
        }

    except requests.RequestException as exc:
        return {
            "success": False,
            "error": f"Could not determine location: {exc}",
        }


def get_location() -> dict[str, Any]:
    """
    Get the user's current city from browser coordinates.

    The browser coordinates are provided by the FastAPI request and stored
    temporarily in a ContextVar before the agent is invoked.
    """
    coordinates = current_coordinates.get()

    if not coordinates:
        return {
            "success": False,
            "error": (
                "Browser location is unavailable. Ask the user to provide "
                "a city or enable location access."
            ),
        }

    return reverse_geocode(
        latitude=coordinates["latitude"],
        longitude=coordinates["longitude"],
    )


SYSTEM_PROMPT = """
You are a helpful weather assistant.

Workflow:

1. If the user asks about weather without specifying a location:
   - Call get_location first.
   - If get_location succeeds, call get_weather using the returned city
     and country.
   - If get_location fails, ask the user to provide a city or enable browser
     location access.

2. If the user provides a city:
   - Call get_weather directly.

3. Temperature units:
   - Use Fahrenheit only for the United States, Liberia, and Myanmar.
   - Use Celsius for every other country.
   - Do not display both units unless the user explicitly requests both.

4. Tool errors:
   - If a tool returns success=false, explain the error clearly.
   - Never invent weather information.

5. Keep the final response concise and friendly.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    connection = sqlite3.connect(
        "checkpoints.db",
        check_same_thread=False,
    )

    checkpointer = SqliteSaver(connection)
    checkpointer.setup()

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.7,
    )

    agent = create_agent(
        model=llm,
        tools=[get_weather, get_location],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    app.state.agent = agent
    app.state.sqlite_connection = connection

    yield

    connection.close()


app = FastAPI(
    title="Weather Assistant",
    lifespan=lifespan,
)

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET_KEY,
    same_site="lax",
    https_only=False,  # Change to True when deployed behind HTTPS.
    max_age=60 * 60 * 24 * 7,
)

templates = Jinja2Templates(directory="templates")


def initialize_session(request: Request) -> None:
    if "thread_id" not in request.session:
        request.session["thread_id"] = str(uuid.uuid4())

    if "messages" not in request.session:
        request.session["messages"] = []


def is_async_request(request: Request) -> bool:
    return request.headers.get("X-Requested-With") == "XMLHttpRequest"


def extract_ai_content(content: Any) -> str:
    """
    Convert Gemini/LangChain content into displayable text.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, dict) and block.get("text"):
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)

        return "\n".join(text_parts)

    return str(content)


@app.get("/")
def home(request: Request):
    initialize_session(request)

    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={
            "messages": request.session["messages"],
        },
    )


@app.post("/send")
def send_message(
    request: Request,
    message: str = Form(default=""),
    latitude: str | None = Form(default=None),
    longitude: str | None = Form(default=None),
):
    initialize_session(request)

    user_message = message.strip()

    if not user_message:
        if is_async_request(request):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Please enter a message.",
                },
            )

        return RedirectResponse(url="/", status_code=303)

    messages = list(request.session.get("messages", []))
    messages.append(
        {
            "sender": "user",
            "text": user_message,
        }
    )

    user_latitude: float | None = None
    user_longitude: float | None = None

    if latitude and longitude:
        try:
            user_latitude = float(latitude)
            user_longitude = float(longitude)

            request.session["user_location"] = {
                "latitude": user_latitude,
                "longitude": user_longitude,
            }
        except ValueError:
            user_latitude = None
            user_longitude = None

    if user_latitude is None or user_longitude is None:
        saved_location = request.session.get("user_location", {})
        user_latitude = saved_location.get("latitude")
        user_longitude = saved_location.get("longitude")

    coordinates_token = None

    if user_latitude is not None and user_longitude is not None:
        coordinates_token = current_coordinates.set(
            {
                "latitude": user_latitude,
                "longitude": user_longitude,
            }
        )

    try:
        response = request.app.state.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_message,
                    }
                ]
            },
            {
                "configurable": {
                    "thread_id": request.session["thread_id"],
                }
            },
        )

        ai_message = response["messages"][-1]
        ai_response = extract_ai_content(ai_message.content)

        if not ai_response:
            ai_response = "I could not generate a response."

    except Exception as exc:
        print(f"Agent error: {exc}")
        ai_response = (
            "Sorry, I encountered an error while processing your request."
        )

    finally:
        if coordinates_token is not None:
            current_coordinates.reset(coordinates_token)

    messages.append(
        {
            "sender": "agent",
            "text": str(ai_response),
        }
    )

    request.session["messages"] = messages

    if is_async_request(request):
        return JSONResponse(
            content={
                "success": True,
                "user_message": user_message,
                "agent_message": str(ai_response),
            }
        )

    return RedirectResponse(url="/", status_code=303)


@app.get("/clear")
def clear_chat(request: Request):
    request.session.clear()
    request.session["thread_id"] = str(uuid.uuid4())
    request.session["messages"] = []

    return RedirectResponse(url="/", status_code=303)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "weather-assistant",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
