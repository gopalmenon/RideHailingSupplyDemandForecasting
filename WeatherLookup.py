import bisect
import FileIo

########################################################################################################################
#                                                                                                                      #
# WeatherSnapshot holds Conditions like sunny, rainy, and snowy etc encoded as integers, temperature in Celsius as a   #
# double and air pollution particulate matter as a double                                                              #
#                                                                                                                      #
########################################################################################################################


class WeatherSnapshot(object):

    """Constructor"""
    def __init__(self, weather_conditions, temperature_in_celsius, particulate_matter):
        self.weather_conditions = weather_conditions
        self.temperature_in_celsius = temperature_in_celsius
        self.particulate_matter = particulate_matter

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

    """Constructor"""
    def __init__(self, data_set):

        """Create storage to hold weather data"""
        self.data_set = data_set
        self.weather_data = dict()

        """Initialize files list based on which environment is being worked on"""
        data_files_path = None
        data_files_list = None
        if self.data_set == WeatherLookup.TRAINING_DATA_SET:
            data_files_path = WeatherLookup.TRAINING_DATA_WEATHER_FILES_PATH
            data_files_list = WeatherLookup.TRAINING_DATA_WEATHER_FILES
        else:
            data_files_path = WeatherLookup.TESTING_DATA_WEATHER_FILES_PATH
            data_files_list = WeatherLookup.TESTING_DATA_WEATHER_FILES

        """Fill lookup dictionary from data files"""
        for file_name in data_files_list:

            """Read the file"""
            text_file_contents = FileIo.get_text_file_contents(data_files_path + file_name)

            """Loop through the records and load the dictionary lookup"""
            for record in text_file_contents:

                record_tokens = record.split(WeatherLookup.TAB_CHARACTER)
                self.weather_data[record_tokens[0]] = WeatherSnapshot(record_tokens[1], record_tokens[2], record_tokens[3].strip())


    """Return weather snapshot closest to the timestamp passsed in"""
    def get_weather_snapshot(self, time_stamp):

        time_stamp_keys = self.weather_data.keys()
        time_stamp_keys_index = bisect.bisect(list(time_stamp_keys), time_stamp)
        return self.weather_data[list(time_stamp_keys)[time_stamp_keys_index]]