import bisect
from datetime import datetime
import FileIo
import time
import unittest
########################################################################################################################
#                                                                                                                      #
# TrafficLookup stores the traffic levels for districts at 10 minute intervals. Given a district hash and a timestamp  #
# the getTraffic method will return traffic at each of the four levels for the nearest timestamp available.            #
#                                                                                                                      #
########################################################################################################################


class TrafficLookup(object):

    TRAFFIC_LEVEL_AND_ROAD_SECTION_SEPARATOR = ":"
    TRAINING_DATA_TRAFFIC_FILES_PATH = "data/season_1/training_data/traffic_data/"
    TRAINING_DATA_TRAFFIC_FILES = ["traffic_data_2016-01-01", "traffic_data_2016-01-02", "traffic_data_2016-01-03",
                                   "traffic_data_2016-01-04", "traffic_data_2016-01-05", "traffic_data_2016-01-06",
                                   "traffic_data_2016-01-07", "traffic_data_2016-01-08", "traffic_data_2016-01-09",
                                   "traffic_data_2016-01-10", "traffic_data_2016-01-11", "traffic_data_2016-01-12",
                                   "traffic_data_2016-01-13", "traffic_data_2016-01-14", "traffic_data_2016-01-15",
                                   "traffic_data_2016-01-16", "traffic_data_2016-01-17", "traffic_data_2016-01-18",
                                   "traffic_data_2016-01-19", "traffic_data_2016-01-20", "traffic_data_2016-01-21"]

    TESTING_DATA_TRAFFIC_FILES_PATH = "data/season_1/test_set_1/traffic_data/"
    TESTING_DATA_TRAFFIC_FILES = ["traffic_data_2016-01-22_test", "traffic_data_2016-01-24_test",
                                  "traffic_data_2016-01-26_test", "traffic_data_2016-01-28_test",
                                  "traffic_data_2016-01-30_test"]

    """
    Constructor
    """
    def __init__(self, data_set):

        # Create storage to hold traffic data
        self.data_set = data_set
        self.traffic_data = list()
        self.traffic_data_keys = None

        # Initialize files list based on which environment is being worked on
        data_files_path = None
        data_files_list = None
        if self.data_set == FileIo.TRAINING_DATA_SET:
            data_files_path = TrafficLookup.TRAINING_DATA_TRAFFIC_FILES_PATH
            data_files_list = TrafficLookup.TRAINING_DATA_TRAFFIC_FILES
        else:
            data_files_path = TrafficLookup.TESTING_DATA_TRAFFIC_FILES_PATH
            data_files_list = TrafficLookup.TESTING_DATA_TRAFFIC_FILES

        # Fill lookup dictionary from data files
        for file_name in data_files_list:

            # Read the file
            text_file_contents = FileIo.get_text_file_contents(data_files_path + file_name)

            # Loop through the records and load the dictionary lookup
            for record in text_file_contents:

                traffic_record = list()
                record_tokens = record.split(FileIo.TAB_CHARACTER)

                # Separate out district hash, traffic at the four congestion levels and timestamp
                traffic_record.append(record_tokens[0])
                traffic_record.\
                    append(int(record_tokens[1].split(TrafficLookup.TRAFFIC_LEVEL_AND_ROAD_SECTION_SEPARATOR)[1]))
                traffic_record.\
                    append(int(record_tokens[2].split(TrafficLookup.TRAFFIC_LEVEL_AND_ROAD_SECTION_SEPARATOR)[1]))
                traffic_record.\
                    append(int(record_tokens[3].split(TrafficLookup.TRAFFIC_LEVEL_AND_ROAD_SECTION_SEPARATOR)[1]))
                traffic_record.\
                    append(int(record_tokens[4].split(TrafficLookup.TRAFFIC_LEVEL_AND_ROAD_SECTION_SEPARATOR)[1]))
                traffic_record.\
                    append(time.mktime(time.strptime(record_tokens[5].strip(), FileIo.TIMESTAMP_FORMAT)))

                self.traffic_data.append(traffic_record)

        # Sort the traffic data so that searching on district hash and timestamp is possible
        self.traffic_data.sort(key = lambda x: (x[0], x[5]))
        self.traffic_data_keys = [[record[0], record[5]] for record in self.traffic_data]

    """
    Return the traffic for the district at the time closest to the timestamp parameter
    """
    def get_road_section_numbers_for_traffic_congestion_levels(self, district_hash, order_time):

        order_timestamp = time.mktime(time.strptime(order_time, FileIo.TIMESTAMP_FORMAT))

        # Find the traffic records before and after the input parameters
        traffic_record_before_index = bisect.bisect_left(self.traffic_data_keys, [district_hash, order_timestamp])
        traffic_record_after_index = bisect.bisect_right(self.traffic_data_keys, [district_hash, order_timestamp])

        # Check for boundary conditions
        if traffic_record_before_index >= len(self.traffic_data_keys):
            traffic_record_before_index = len(self.traffic_data_keys) - 1
        if traffic_record_after_index >= len(self.traffic_data_keys):
            traffic_record_after_index = len(self.traffic_data_keys) - 1
        if traffic_record_before_index == traffic_record_after_index and traffic_record_after_index != 0:
            traffic_record_before_index -= 1

        # Check if before and after records are for same district hash and use the one with the closest timestamp
        if self.traffic_data[traffic_record_before_index][0] == district_hash:
            if self.traffic_data[traffic_record_after_index][0] == district_hash:
                date_time_before = self.traffic_data_keys[traffic_record_before_index][1]
                date_time_after = self.traffic_data_keys[traffic_record_after_index][1]
                if abs(order_timestamp - date_time_before) > abs(date_time_after - order_timestamp):
                    return self.__get_traffic_conditions(traffic_record_after_index)
                else:
                    return self.__get_traffic_conditions(traffic_record_before_index)
            else:
                return self.__get_traffic_conditions(traffic_record_before_index)
        else:
            if self.traffic_data[traffic_record_after_index][0] == district_hash:
                return self.__get_traffic_conditions(traffic_record_after_index)
            else:
                return None

    """
    Return traffic conditions corresponding to the input index
    """
    def __get_traffic_conditions(self, index):

        return [self.traffic_data[index][1], self.traffic_data[index][2],
                self.traffic_data[index][3], self.traffic_data[index][4]]

########################################################################################################################
#                                                                                                                      #
# Unit testing class.                                                                                                  #
#                                                                                                                      #
########################################################################################################################


class TestTrafficLookup(unittest.TestCase):

    def test_before_range(self):
        traffic_lookup = TrafficLookup("test")
        roads_at_congestion_levels = traffic_lookup.get_road_section_numbers_for_traffic_congestion_levels("58c7a4888306d8ff3a641d1c0feccbe3", "2016-01-22 06:10:29")
        self.assertEquals(roads_at_congestion_levels, [182, 31, 10, 3])

    def test_exact_match(self):
        traffic_lookup = TrafficLookup("test")
        roads_at_congestion_levels = traffic_lookup.get_road_section_numbers_for_traffic_congestion_levels("cb6041cc08444746caf6039d8b9e43cb", "2016-01-26 07:10:29")
        self.assertEquals(roads_at_congestion_levels, [138, 27, 2, 7])

    def test_middle_of_range(self):
        traffic_lookup = TrafficLookup("test")
        roads_at_congestion_levels = traffic_lookup.get_road_section_numbers_for_traffic_congestion_levels("4725c39a5e5f4c188d382da3910b3f3f", "2016-01-26 12:00:00")
        self.assertEquals(roads_at_congestion_levels, [2707, 844, 383, 270])

    def test_after_range(self):
        traffic_lookup = TrafficLookup("test")
        roads_at_congestion_levels = traffic_lookup.get_road_section_numbers_for_traffic_congestion_levels("44c097b7bd219d104050abbafe51bd49", "2016-01-31 23:55:26")
        self.assertEquals(roads_at_congestion_levels, [257, 15, 1, 4])

"""
Self test
"""
if __name__ == "__main__":
    unittest.main()