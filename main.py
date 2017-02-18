import src.Weather.WeatherLookup as WeatherLookup
from src.POI.POI import ReadPOI

weather_lookup = WeatherLookup.WeatherLookup("test")
print(weather_lookup.get_weather_snapshot("2016-01-24 15:21:53"))

poi = ReadPOI('data/season_1/test_set_1/poi_data/poi_data').readFile()
for x,v in poi.items():
    print (x,v)