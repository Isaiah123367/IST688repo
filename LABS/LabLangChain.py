import os
import requests
import streamlit as st
from langchain.agents import create_agent

# ---------------------------------------------------------------------
# Lab 06 - The "What to Wear" Bot, rebuilt with LangChain
# ---------------------------------------------------------------------
st.title("👕 What to Wear Bot (LangChain)")
st.write(
    "Enter a city and I'll check the weather, then suggest what to wear "
    "today and some outdoor activities that suit the conditions. "
    "Leave it blank to use the default location (Syracuse, NY)."
)

DEFAULT_LOCATION = "Syracuse, NY"

# LangChain's OpenAI integration reads the key from the environment
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# ---------------------------------------------------------------------
# Part B: the tool. The docstring is sent to the model as the tool
# description, so it is part of the prompt.
# ---------------------------------------------------------------------
def get_weather_data(location: str = DEFAULT_LOCATION) -> dict:
    """Get current conditions and today's forecast for a location: current
    temperature and feels-like (F), humidity, wind (mph), precipitation, UV,
    today's high/low, sunrise/sunset, and hourly temperature, rain/snow
    chances, wind and UV from 6 AM to 9 PM. Accepts a city and state/country
    (e.g. 'Syracuse, NY', 'Lima, Peru'), a zip code, an airport code ('SYR'),
    or a landmark ('Eiffel Tower'). Call it once per location. If the user
    gave no location, use Syracuse, NY. Returns an 'error' key if the place
    can't be found.
    """
    try:
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
    except Exception as e:
        # Return the error as data so the model can explain it (same as Lab 5)
        return {"error": str(e), "requested_location": location}


# ---------------------------------------------------------------------
# Part C: the agent (system prompt reused from Lab 5, plus the Lab 5
# second-call instructions, since there is no second call to hold them)
# ---------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a friendly, practical assistant that helps people decide what to "
    "wear and what to do outside based on the weather. When you need weather "
    "data, call the get_weather_data tool. If the user gives no location, "
    "call the tool without a location (it defaults to Syracuse, NY). "
    "After getting the weather, reply with: "
    "1. Suggestions on appropriate clothes to wear today, accounting for how "
    "conditions change through the day (layers, rain gear, sun protection, "
    "etc.). "
    "2. Suggestions for outdoor activities appropriate to today's weather, "
    "including the best time of day for them. "
    "Start by naming the location and briefly summarizing the weather. "
    "If there was an error getting the weather, explain it and ask the user "
    "to try a different location."
)

agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[get_weather_data],
    system_prompt=SYSTEM_PROMPT,
)

city = st.text_input("City", placeholder="e.g. Syracuse, NY or Lima, Peru")

if st.button("Get advice", type="primary"):
    if city.strip():
        user_request = f"What should I wear today in {city.strip()}, and what outdoor activities would be good there today?"
    else:
        user_request = "What should I wear today, and what outdoor activities would be good today?"

    with st.spinner("Checking the weather..."):
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_request}]}
        )

    with st.chat_message("assistant"):
        st.write(result["messages"][-1].content)

    # -----------------------------------------------------------------
    # Part D: look inside the agent
    # -----------------------------------------------------------------
    with st.expander("Behind the scenes"):
        for message in result["messages"]:
            st.write(f"**{message.type}**")
            st.write(message.content)
            if message.type == "ai" and message.tool_calls:
                st.write(message.tool_calls)


# ---------------------------------------------------------------------
# Part D answers (for a single city, e.g. "Syracuse, NY")
#
# a. 4 messages:
#    1) human - the user request ("What should I wear today in Syracuse, NY...")
#    2) ai    - empty content; it only carries the request to call the tool
#    3) tool  - the weather dict returned by get_weather_data
#    4) ai    - the final clothing/activity advice
#    (The system prompt is passed separately and is not in result["messages"].)
#
# b. Message 2 (the first ai message). The request is not in .content but in
#    its .tool_calls list: name="get_weather_data", args={"location": ...},
#    and an id that the tool message refers back to.
#
# c. In Lab 5 (Lab5.py):
#    1) human message  -> lines 150-153, the `messages = [...]` list
#    2) ai tool-call   -> lines 157-163, the first client.chat.completions.create
#                         call (first_msg), appended on line 169
#    3) tool message   -> lines 185-189, messages.append({"role": "tool",
#                         "tool_call_id": ..., "content": json.dumps(weather)})
#                         after calling get_current_weather on line 178
#    4) final ai msg   -> lines 207-212, the second create(..., stream=True)
#                         call (plus the extra user message on lines 192-205)
#
# Part E / F answers are in Submission_Answers.md.
# ---------------------------------------------------------------------
