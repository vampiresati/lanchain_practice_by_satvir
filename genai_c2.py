import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import create_agent
import os

load_dotenv()


def get_weather(city: str):
    """
    Get live weather for a given city.
    Returns both Celsius and Fahrenheit temperatures.
    """

    # Step 1: Get latitude and longitude
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

    if "results" not in geo_data or not geo_data["results"]:
        return {"error": f"City '{city}' not found."}

    location = geo_data["results"][0]

    latitude = location["latitude"]
    longitude = location["longitude"]

    # Step 2: Get current weather
    weather_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "timezone": "auto",
        },
        timeout=10,
    )

    weather_response.raise_for_status()
    weather_data = weather_response.json()

    current = weather_data["current"]

    temperature_celsius = current["temperature_2m"]
    temperature_fahrenheit = round((temperature_celsius * 9 / 5) + 32, 1)

    return {
        "city": location["name"],
        "country": location["country"],
        "temperature_celsius": temperature_celsius,
        "temperature_fahrenheit": temperature_fahrenheit,
        "humidity": current["relative_humidity_2m"],
        "wind_speed_kmh": current["wind_speed_10m"],
        "weather_code": current["weather_code"],
    }

def get_location():
    """Get user's current location. Use this when the user asks about weather."""
    response = requests.get("https://ipapi.co/json/", headers={'User-agent': 'your-bot 0.1'})
    data = response.json()
    city = data['city']
    country = data.get('country_name')
    return f"{city}, {country}"


# Initialize Gemini Flash 2.5
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.7,
)
system_prompt = """
You are a helpful weather assistant. 
YOUR WORKFLOW:
1. If the user asks about weather WITHOUT specifying a location, you MUST:
   - First call get_location() to find their location
   - Then call get_weather(city) with that location

2. If the user provides a city, call get_weather(city) directly.

3. Use your knowledge to determine which temperature unit is standard for the given location.

4. Present the weather information including temperature, condition, wind speed, and any other relevant details.

"""
agent = create_agent(
    model=llm,
    tools=[get_weather, get_location],
    system_prompt=system_prompt
)

if __name__ == "__main__":
    user_query = input("Enter your query: ")

    # response1 = llm.invoke("How is the weather in Rome?")
    response1 = agent.invoke(
        {"messages": [{'role': 'user',
                       'content': user_query}]})
    print(response1['messages'][-1].content)

#
# print(get_location())
# print(get_weather("Bathinda"))



# def get_weather(city: str):
#     """Get weather for a given city.
#     Return the temperature_fahrenheit value in Fahrenheit label for locations such as US, Liberia, Burma"""
#     api_key = os.getenv("OPENWEATHER_API_KEY")
#     base_url = "http://api.openweathermap.org/data/2.5/weather"
#     params = {
#         "q": city,
#         "appid": api_key,
#         'units': 'metric'
#     }
#     response = requests.get(base_url, params=params)
#     data = response.json()
#     print(data)
#     temperature_celsius = data['main']['temp']
#     temperature_fahrenheit = temperature_celsius * 9 / 5 + 32
#     return data, {'temperature_fahrenheit': temperature_fahrenheit}
