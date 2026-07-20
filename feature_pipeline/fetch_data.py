import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

# ==============================
# Load Environment Variables
# ==============================
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not API_KEY:
    raise ValueError("❌ OPENWEATHER_API_KEY not found in .env file")

# ==============================
# Configuration
# ==============================
CITY = "Karachi"
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
AIR_POLLUTION_URL = "https://api.openweathermap.org/data/2.5/air_pollution"

# Create raw data folder if it doesn't exist
RAW_DATA_DIR = "data/raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Timestamp for filenames
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# ==============================
# Function to Save JSON
# ==============================
def save_json(data, filename):
    filepath = os.path.join(RAW_DATA_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    print(f"✅ Saved: {filepath}")


# ==============================
# Fetch Weather Data
# ==============================
print(f"\n🌤 Fetching weather data for {CITY}...")

weather_params = {
    "q": CITY,
    "appid": API_KEY,
    "units": "metric"
}

try:
    weather_response = requests.get(
        WEATHER_URL,
        params=weather_params,
        timeout=10
    )

    weather_response.raise_for_status()

    weather_data = weather_response.json()

    save_json(
        weather_data,
        f"weather_{timestamp}.json"
    )

except requests.exceptions.RequestException as e:
    print("❌ Error fetching weather data")
    print(e)
    exit()

# ==============================
# Extract Coordinates
# ==============================
lat = weather_data["coord"]["lat"]
lon = weather_data["coord"]["lon"]

print(f"\n📍 Coordinates")
print(f"Latitude : {lat}")
print(f"Longitude: {lon}")

# ==============================
# Fetch AQI Data
# ==============================
print("\n🌫 Fetching AQI data...")

aqi_params = {
    "lat": lat,
    "lon": lon,
    "appid": API_KEY
}

try:
    aqi_response = requests.get(
        AIR_POLLUTION_URL,
        params=aqi_params,
        timeout=10
    )

    aqi_response.raise_for_status()

    aqi_data = aqi_response.json()

    save_json(
        aqi_data,
        f"aqi_{timestamp}.json"
    )

except requests.exceptions.RequestException as e:
    print("❌ Error fetching AQI data")
    print(e)
    exit()

# ==============================
# Print Summary
# ==============================
weather = weather_data["weather"][0]

main = weather_data["main"]

wind = weather_data["wind"]

pollution = aqi_data["list"][0]

components = pollution["components"]

print("\n==============================")
print("      DATA COLLECTION SUCCESS")
print("==============================")

print(f"City            : {CITY}")
print(f"Temperature     : {main['temp']} °C")
print(f"Feels Like      : {main['feels_like']} °C")
print(f"Humidity        : {main['humidity']} %")
print(f"Pressure        : {main['pressure']} hPa")
print(f"Wind Speed      : {wind['speed']} m/s")
print(f"Weather         : {weather['description']}")

print("\n----- Air Pollution -----")
print(f"AQI             : {pollution['main']['aqi']}")
print(f"PM2.5           : {components['pm2_5']}")
print(f"PM10            : {components['pm10']}")
print(f"CO              : {components['co']}")
print(f"NO₂             : {components['no2']}")
print(f"SO₂             : {components['so2']}")
print(f"O₃              : {components['o3']}")
print(f"NH₃             : {components['nh3']}")

print("\nDay 1 data collection completed successfully!")