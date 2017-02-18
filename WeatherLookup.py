import bisect
import time
import unittest
from collections import OrderedDict
from datetime import datetime

import FileIo as FileIo


########################################################################################################################
#                                                                                                                      #
# WeatherSnapshot holds Conditions like sunny, rainy, and snowy etc encoded as integers, temperature in Celsius as a   #
# double and air pollution particulate matter as a double                                                              #
#                                                                                                                      #
########################################################################################################################


class WeatherSnapshot(object):

    """Constructor"""
    def __init__(self, weather_conditions, temperature_in_celsius, particulate_matter):
        self.weather_conditions = int(weather_conditions)
        self.temperature_in_celsius = float(temperature_in_celsius)
        self.particulate_matter = float(particulate_matter)

    def get_weather_conditions(self):
        return self.weather_conditions

    def get_temperature_in_celsius(self):
        return self.temperature_in_celsius

    def get_particulate_matter(self):
        return self.particulate_matter

    def __str__(self):
        return "Weather Condition: " + str(self.weather_conditions) + ", Temperature in Celsius: " + \
               str(self.temperature_in_celsius) + ", Particulate Matter: " + str(self.particulate_matter)

########################################################################################################################
#                                                                                                                      #
# WeatherLookup returns the weather snapshot given a timestamp.                                                        #
#                                                                                                                      #
########################################################################################################################


class WeatherLookup(object):

    TAB_CHARACTER = "\t"

    TRAINING_DATA_SET = "train"
    TEST_DATA_SET = "test"

    TRAINING_DATA_WEATHER_FILES_PATH = "data/season_1/training_data/weather_data/"
    TRAINING_DATA_WEATHER_FILES = ["weather_data_2016-01-01", "weather_data_2016-01-02", "weather_data_2016-01-03",
                                   "weather_data_2016-01-04", "weather_data_2016-01-05", "weather_data_2016-01-06",
                                   "weather_data_2016-01-07", "weather_data_2016-01-08", "weather_data_2016-01-09",
                                   "weather_data_2016-01-10", "weather_data_2016-01-11", "weather_data_2016-01-12",
                                   "weather_data_2016-01-13", "weather_data_2016-01-14", "weather_data_2016-01-15",
                                   "weather_data_2016-01-16", "weather_data_2016-01-17", "weather_data_2016-01-18",
                                   "weather_data_2016-01-19", "weather_data_2016-01-20", "weather_data_2016-01-21"]

    TESTING_DATA_WEATHER_FILES_PATH = "data/season_1/test_set_1/weather_data/"
    TESTING_DATA_WEATHER_FILES = ["weather_data_2016-01-22_test", "weather_data_2016-01-24_test",
                                  "weather_data_2016-01-26_test", "weather_data_2016-01-28_test",
                                  "weather_data_2016-01-30_test"]

    """
    Constructor
    """
    def __init__(self, data_set):

        # Create storage to hold weather data
        self.data_set = data_set
        self.weather_data = None

        # Initialize files list based on which environment is being worked on
        data_files_path = None
        data_files_list = None
        if self.data_set == WeatherLookup.TRAINING_DATA_SET:
            data_files_path = WeatherLookup.TRAINING_DATA_WEATHER_FILES_PATH
            data_files_list = WeatherLookup.TRAINING_DATA_WEATHER_FILES
        else:
            data_files_path = WeatherLookup.TESTING_DATA_WEATHER_FILES_PATH
            data_files_list = WeatherLookup.TESTING_DATA_WEATHER_FILES

        unsorted_weather_data = dict()
        # Fill lookup dictionary from data files
        for file_name in data_files_list:

            # Read the file
            text_file_contents = FileIo.get_text_file_contents(data_files_path + file_name)

            # Loop through the records and load the dictionary lookup
            for record in text_file_contents:

                record_tokens = record.split(WeatherLookup.TAB_CHARACTER)
                unsorted_weather_data[record_tokens[0]] \
                    = WeatherSnapshot(record_tokens[1], record_tokens[2], record_tokens[3].strip())

        # Sort the weather data so that searching on timestamp is possible
        self.weather_data = OrderedDict(sorted(unsorted_weather_data.items(),
                                               key = lambda x: time.mktime(time.strptime(x[0], FileIo.TIMESTAMP_FORMAT))))

    """
    Return weather snapshot closest to the timestamp parameter
    """
    def get_weather_snapshot(self, time_stamp):

        # find the closest timestamps before and after the input parameter
        time_stamp_keys = self.weather_data.keys()
        time_stamp_before_index = bisect.bisect_left(list(time_stamp_keys), time_stamp)
        time_stamp_after_index = bisect.bisect_right(list(time_stamp_keys), time_stamp)
        if time_stamp_before_index == time_stamp_after_index and time_stamp_before_index != 0:
            time_stamp_before_index -= 1

        # Convert timestamps to date time values for easy comparison
        if time_stamp_before_index >= len(time_stamp_keys):
            time_stamp_before_index = len(time_stamp_keys) -1
        if time_stamp_after_index >= len(time_stamp_keys):
            time_stamp_after_index = len(time_stamp_keys) -1
        date_time_after = datetime.strptime(list(time_stamp_keys)[time_stamp_after_index], FileIo.TIMESTAMP_FORMAT)
        date_time_before = datetime.strptime(list(time_stamp_keys)[time_stamp_before_index], FileIo.TIMESTAMP_FORMAT)
        date_time_input = datetime.strptime(time_stamp, FileIo.TIMESTAMP_FORMAT)

        # Find out which one is closest to the input parameter
        closest_index = None
        if abs(date_time_before - date_time_input) > abs(date_time_input - date_time_after):
            closest_index = time_stamp_after_index
        elif abs(date_time_before - date_time_input) < abs(date_time_input - date_time_after):
            closest_index = time_stamp_before_index
        else:
            closest_index = time_stamp_before_index

        # Return the weather corresponding to the entry with timestamp closest to the input timestamp parameter
        return self.weather_data[list(time_stamp_keys)[closest_index]]


########################################################################################################################
#                                                                                                                      #
# Unit testing class.                                                                                                  #
#                                                                                                                      #
########################################################################################################################


class TestWeatherLookup(unittest.TestCase):

    def test_before_range(self):
        weather_lookup = WeatherLookup("test")
        weather = weather_lookup.get_weather_snapshot("2016-01-15 06:12:22")
        self.assertEquals(weather.get_weather_conditions(), 2) and \
            self.assertEquals(weather.get_temperature_in_celsius(), 2.0) and \
            self.assertEquals(weather.get_particulate_matter(), 51)

    def test_start_of_range(self):
        weather_lookup = WeatherLookup("test")
        weather = weather_lookup.get_weather_snapshot("2016-01-22 07:02:10")
        self.assertEquals(weather.get_weather_conditions(), 2) and \
            self.assertEquals(weather.get_temperature_in_celsius(), 2.0) and \
            self.assertEquals(weather.get_particulate_matter(), 51)

    def test_between_files(self):
        weather_lookup = WeatherLookup("test")
        weather = weather_lookup.get_weather_snapshot("2016-01-23 01:15:15")
        self.assertEquals(weather.get_weather_conditions(), 6) and \
            self.assertEquals(weather.get_temperature_in_celsius(), 1.0) and \
            self.assertEquals(weather.get_particulate_matter(), 98)

    def test_exact_match(self):
        weather_lookup = WeatherLookup("test")
        weather = weather_lookup.get_weather_snapshot("2016-01-28 15:21:37")
        self.assertEquals(weather.get_weather_conditions(), 4) and \
            self.assertEquals(weather.get_temperature_in_celsius(), 5.0) and \
            self.assertEquals(weather.get_particulate_matter(), 79)

    def test_end_of_range(self):
        weather_lookup = WeatherLookup("test")
        weather = weather_lookup.get_weather_snapshot("2016-01-30 23:20:30")
        self.assertEquals(weather.get_weather_conditions(), 3) and \
            self.assertEquals(weather.get_temperature_in_celsius(), 3.0) and \
            self.assertEquals(weather.get_particulate_matter(), 68)

    def test_after_range(self):
        weather_lookup = WeatherLookup("test")
        weather = weather_lookup.get_weather_snapshot("2016-01-31 12:35:23")
        self.assertEquals(weather.get_weather_conditions(), 3) and \
            self.assertEquals(weather.get_temperature_in_celsius(), 3.0) and \
            self.assertEquals(weather.get_particulate_matter(), 68)

"""
Self test
"""
if __name__ == "__main__":
    unittest.main()
