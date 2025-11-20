import requests
from datetime import datetime

# -------------------------------
# Yr.no (Free, no API key needed)
# -------------------------------
def get_forecast_yr(latitude, longitude):
    url = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={latitude}&lon={longitude}"
    headers = {
        "User-Agent": "Mike3.0-Bot/1.0"
    }
    response = requests.get(url, headers=headers)
    data = response.json()
    forecast = []
    for t in data["properties"]["timeseries"][:6]:  # next 6 hours
        dt = t["time"]
        temp = t["data"]["instant"]["details"]["air_temperature"]
        forecast.append((dt, temp))
    return forecast

# -------------------------------
# AccuWeather (Requires API Key)
# -------------------------------
def get_forecast_accuweather(api_key, location_key):
    url = f"http://dataservice.accuweather.com/forecasts/v1/hourly/12hour/{location_key}?apikey={api_key}&metric=true"
    response = requests.get(url)
    data = response.json()
    forecast = []
    for t in data[:6]:  # next 6 hours
        dt = t["DateTime"]
        temp = t["Temperature"]["Value"]
        forecast.append((dt, temp))
    return forecast

# -------------------------------
# Weather.com / The Weather Channel (Requires API Key)
# -------------------------------
def get_forecast_weather_com(api_key, latitude, longitude):
    url = f"https://api.weather.com/v3/wx/forecast/hourly/12hour?apiKey={api_key}&geocode={latitude},{longitude}&format=json&units=m"
    response = requests.get(url)
    data = response.json()
    forecast = []
    for dt, temp in zip(data["validTimeLocal"][:6], data["temperature"][:6]):
        forecast.append((dt, temp))
    return forecast

# -------------------------------
# GraphCast / Aurora placeholder
# -------------------------------
def get_forecast_graphcast(latitude, longitude):
    """
    These are ML-based forecasts, normally not public.
    Here we just simulate with a placeholder.
    """
    now = datetime.utcnow()
    forecast = [(now, 20.0)]  # dummy temperature
    return forecast

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    lat, lon = 40.7128, -74.0060  # New York City
    accu_api_key = "YOUR_ACCUWEATHER_KEY"
    accu_location_key = "349727"  # NYC location key
    weather_com_api_key = "YOUR_WEATHER_COM_KEY"

    print("Yr.no forecast:", get_forecast_yr(lat, lon))
    print("AccuWeather forecast:", get_forecast_accuweather(accu_api_key, accu_location_key))
    print("Weather.com forecast:", get_forecast_weather_com(weather_com_api_key, lat, lon))
    print("GraphCast forecast (simulated):", get_forecast_graphcast(lat, lon))
