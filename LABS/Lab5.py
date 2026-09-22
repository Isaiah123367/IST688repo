import json
import requests
import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------------
# Lab 05 - The "What to Wear" Bot
# ---------------------------------------------------------------------
st.title("👕 What to Wear Bot")
st.write(
    "Enter a city and I'll check the weather, then suggest what to wear "
    "today and some outdoor activities that suit the conditions. "
    "Leave it blank to use the default location (Syracuse, NY)."
)

DEFAULT_LOCATION = "Syracuse, NY"
MODEL = "gpt-4o-mini"

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# ---------------------------------------------------------------------
# Part A / B: Weather data function (wttr.in, no API key needed)
# ---------------------------------------------------------------------
def get_current_weather(location=DEFAULT_LOCATION):
    """Return the weather details the bot needs for clothing/activity advice.

    location can be a city, a zip code, an airport code ('SYR'),
    or a landmark ('Eiffel+Tower'). Units are hard-coded to Fahrenheit/mph.
    """
    if not location or not str(location).strip():
        location = DEFAULT_LOCATION
    location = str(location).strip()

    url = f"https://wttr.in/{location.replace(' ', '+')}?format=j1"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise Exception(f"wttr.in error: status {response.status_code}")
    try:
        data = response.json()
    except ValueError:
        # unknown locations come back as plain text, not JSON
        raise Exception(f"Could not find a location named {location}")

    current = data["current_condition"][0]
    today = data["weather"][0]
    area = data["nearest_area"][0]
    astronomy = today["astronomy"][0]

    # Hourly forecast (wttr.in gives 3-hour slots; times are local, e.g. "900")
    # Keep the daytime/evening slots so the LLM can see how the day changes.
    wanted_slots = {"600": "6 AM", "900": "9 AM", "1200": "12 PM",
                    "1500": "3 PM", "1800": "6 PM", "2100": "9 PM"}
    hourly = []
    for hour in today["hourly"]:
        if hour["time"] in wanted_slots:
            hourly.append({
                "time": wanted_slots[hour["time"]],
                "temperature_F": float(hour["tempF"]),
                "feels_like_F": float(hour["FeelsLikeF"]),
                "description": hour["weatherDesc"][0]["value"],
                "chance_of_rain_pct": int(hour["chanceofrain"]),
                "chance_of_snow_pct": int(hour["chanceofsnow"]),
                "wind_mph": float(hour["windspeedMiles"]),
                "uv_index": int(hour["uvIndex"]),
            })

    return {
        "requested_location": location,
        "matched_location": ", ".join(
            part for part in [
                area["areaName"][0]["value"],
                area["region"][0]["value"],
                area["country"][0]["value"],
            ] if part
        ),
        "current": {
            "temperature_F": float(current["temp_F"]),
            "feels_like_F": float(current["FeelsLikeF"]),
            "description": current["weatherDesc"][0]["value"],
            "humidity_pct": int(current["humidity"]),
            "wind_mph": float(current["windspeedMiles"]),
            "wind_direction": current["winddir16Point"],
            "precipitation_in": float(current["precipInches"]),
            "uv_index": int(current["uvIndex"]),
            "visibility_miles": float(current["visibilityMiles"]),
        },
        "today": {
            "high_F": float(today["maxtempF"]),
            "low_F": float(today["mintempF"]),
            "sunrise": astronomy["sunrise"],
            "sunset": astronomy["sunset"],
            "hourly": hourly,
        },
    }


# ---------------------------------------------------------------------
# Tool definition for the OpenAI API
# ---------------------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": (
                "Get current conditions and today's forecast (high/low, hourly "
                "temperature, rain/snow chances, wind, UV, sunrise/sunset) for a "
                "location. Use this whenever weather information is needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": (
                            "City and state/country, e.g. 'Syracuse, NY' or "
                            "'Lima, Peru'. Also accepts zip codes or airport codes. "
                            "Omit if the user did not give a location."
                        ),
                    }
                },
                "required": [],
            },
        },
    }
]

SYSTEM_PROMPT = (
    "You are a friendly, practical assistant that helps people decide what to "
    "wear and what to do outside based on the weather. When you need weather "
    "data, call the get_current_weather tool. If the user gives no location, "
    "call the tool without a location (it defaults to Syracuse, NY)."
)


# ---------------------------------------------------------------------
# UI: user enters a city, bot returns advice (not a chatbot)
# ---------------------------------------------------------------------
city = st.text_input("City", placeholder="e.g. Syracuse, NY or Lima, Peru")

if st.button("Get advice", type="primary"):
    if city.strip():
        user_request = (f"What should I wear today in {city.strip()}, and what "
                        f"outdoor activities would be good there today?")
    else:
        user_request = ("What should I wear today, and what outdoor activities "
                        "would be good today?")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_request},
    ]

    # --- First call: let the model decide whether it needs the weather tool
    with st.spinner("Checking the weather..."):
        first = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
    first_msg = first.choices[0].message

    if not first_msg.tool_calls:
        # Model answered without needing the weather
        st.write(first_msg.content)
    else:
        messages.append(first_msg)  # assistant message containing the tool call(s)

        for tool_call in first_msg.tool_calls:
            if tool_call.function.name != "get_current_weather":
                continue
            args = json.loads(tool_call.function.arguments or "{}")
            location = args.get("location") or DEFAULT_LOCATION

            try:
                weather = get_current_weather(location)
                with st.expander(f"Weather data for {weather['matched_location']}"):
                    st.json(weather)
            except Exception as e:
                weather = {"error": str(e), "requested_location": location}
                st.error(str(e))

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(weather),
            })

        # --- Second call: weather info is in the prompt; ask for the advice
        messages.append({
            "role": "user",
            "content": (
                "Using the weather information above, give me:\n"
                "1. Suggestions on appropriate clothes to wear today, accounting "
                "for how conditions change through the day (layers, rain gear, "
                "sun protection, etc.).\n"
                "2. Suggestions for outdoor activities appropriate to today's "
                "weather, including the best time of day for them.\n"
                "Start by naming the location and briefly summarizing the weather. "
                "If there was an error getting the weather, explain it and ask me "
                "to try a different location."
            ),
        })

        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True,
        )
        st.write_stream(stream)
