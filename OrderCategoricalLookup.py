import unittest
import FileIo as FileIo
from datetime import datetime
import math
import unittest

########################################################################################################################
#                                                                                                                      #
# This class stores all possible values for order fields and can return a categorical list for passenger id, driver id,#
# district and order time.                                                                                             #
#                                                                                                                      #
# W A R N I N G : I N S T A N T I A T E  C L A S S  O N L Y  O N C E                                                   #
#                                                                                                                      #
########################################################################################################################


class OrderCategoricalLookup(object):

    TRAINING_DATA_ORDER_FILES_PATH = "data/season_1/training_data/order_data/"
    TRAINING_DATA_ORDER_FILES = ["order_data_2016-01-01", "order_data_2016-01-02", "order_data_2016-01-03",
                                 "order_data_2016-01-04", "order_data_2016-01-05", "order_data_2016-01-06",
                                 "order_data_2016-01-07", "order_data_2016-01-08", "order_data_2016-01-09",
                                 "order_data_2016-01-10", "order_data_2016-01-11", "order_data_2016-01-12",
                                 "order_data_2016-01-13", "order_data_2016-01-14", "order_data_2016-01-15",
                                 "order_data_2016-01-16", "order_data_2016-01-17", "order_data_2016-01-18",
                                 "order_data_2016-01-19", "order_data_2016-01-20", "order_data_2016-01-21"]

    TESTING_DATA_ORDER_FILES_PATH = "data/season_1/test_set_1/order_data/"
    TESTING_DATA_ORDER_FILES = ["order_data_2016-01-22_test", "order_data_2016-01-24_test",
                                "order_data_2016-01-26_test", "order_data_2016-01-28_test",
                                "order_data_2016-01-30_test"]

    CHINESE_JANUARY_PUBLIC_HOLIDAYS =[datetime.strptime("2016-01-01", '%Y-%m-%d')]

    DAYS_IN_WEEK = 7
    HOLIDAY_SLOTS = 1
    TEN_MINUTE_SLOTS_IN_A_DAY = 144
    TIMESTAMP_ROW_LENGTH = DAYS_IN_WEEK + HOLIDAY_SLOTS + TEN_MINUTE_SLOTS_IN_A_DAY
    TIMESLOT_OFFSET = DAYS_IN_WEEK + HOLIDAY_SLOTS

    """
    Constructor
    """
    def __init__(self):

        # Create storage for categorical data
        self.driver_ids = list()
        self.passenger_ids = list()
        self.district_hashes = list()

        unique_driver_ids = set()
        unique_passenger_ids = set()
        unique_district_hashes = set()

        # Store all unique occurrences of categorical fields
        for data_set in [FileIo.TRAINING_DATA_SET, FileIo.TEST_DATA_SET]:

            if data_set == FileIo.TRAINING_DATA_SET:
                data_files_path = OrderCategoricalLookup.TRAINING_DATA_ORDER_FILES_PATH
                data_files_list = OrderCategoricalLookup.TRAINING_DATA_ORDER_FILES
            else:
                data_files_path = OrderCategoricalLookup.TESTING_DATA_ORDER_FILES_PATH
                data_files_list = OrderCategoricalLookup.TESTING_DATA_ORDER_FILES

            for file_name in data_files_list:

                # Read the file
                text_file_contents = FileIo.get_text_file_contents(data_files_path + file_name)

                # Loop through the records and load the dictionary lookup
                for record in text_file_contents:
                    record_tokens = record.split(FileIo.TAB_CHARACTER)
                    unique_driver_ids.add(record_tokens[1])
                    unique_passenger_ids.add(record_tokens[2])
                    unique_district_hashes.add(record_tokens[3])
                    unique_district_hashes.add(record_tokens[4])

        # Store unique categorical field values
        self.driver_ids = list(unique_driver_ids)
        self.driver_ids.sort()
        self.passenger_ids = list(unique_passenger_ids)
        self.passenger_ids.sort()
        self.district_hashes = list(unique_district_hashes)
        self.district_hashes.sort()

    """
    Return a list corresponding to a driver id
    """
    def get_driver_id_row(self, driver_id):

        driver_id_row = [0] * len(self.driver_ids)
        driver_id_row[self.driver_ids.index(driver_id)] = 1
        return driver_id_row

    """
    Return a list corresponding to a passenger id
    """
    def get_passenger_id_row(self, passenger_id):

        passenger_id_row = [0] * len(self.passenger_ids)
        passenger_id_row[self.passenger_ids.index(passenger_id)] = 1
        return passenger_id_row


    """
    Return a list corresponding to a district hash
    """
    def get_district_hash_row(self, district_hash):

        district_hash_row = [0] * len(self.district_hashes)
        district_hash_row[self.district_hashes.index(district_hash)] = 1
        return district_hash_row

    """
    Return a list corresponding to a timestamp
    """
    def get_timestamp_row(self, order_timestamp):

        # Create a timestamp row with the first 7 slots for day of the week, the 8th one for holiday yes/no, followed
        # by 144 slots for the 144 parts that a day is divided into.
        timestamp_row = [0] * OrderCategoricalLookup.TIMESTAMP_ROW_LENGTH

        # Get day of week corresponding to order date
        order_datetime = datetime.strptime(order_timestamp, FileIo.TIMESTAMP_FORMAT)
        timestamp_row[order_datetime.weekday()] = 1

        # Check if order date is for a holiday
        if order_datetime.replace(hour=0, minute=0, second=0, microsecond=0) in \
                OrderCategoricalLookup.CHINESE_JANUARY_PUBLIC_HOLIDAYS:
            timestamp_row[7] = 1

        # Each 24 hour day is divided up into 10 minute timeslots. Find timeslot that the order timestamp lies in.
        seconds_since_order_date_midnight = \
            (order_datetime - order_datetime.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
        ten_minute_steps_since_order_date_midnight = int(math.ceil(seconds_since_order_date_midnight / (60 * 10)))
        timestamp_row[OrderCategoricalLookup.TIMESLOT_OFFSET + ten_minute_steps_since_order_date_midnight - 1] = 1

        return timestamp_row

########################################################################################################################
#                                                                                                                      #
# Unit testing class.                                                                                                  #
#                                                                                                                      #
########################################################################################################################


class TestOrderCategoricalLookup(unittest.TestCase):

    def test_fifth_timeslot_new_year_day(self):
        order_categorical_lookup = OrderCategoricalLookup()
        timestamp_row = order_categorical_lookup.get_timestamp_row('2016-01-01 00:44:00')
        expected_timestamp_row = [0] * OrderCategoricalLookup.TIMESTAMP_ROW_LENGTH
        expected_timestamp_row[4] = 1 # Its a Friday. Zero offset corresponds to Monday.
        expected_timestamp_row[7] = 1 # Its a Holiday
        expected_timestamp_row[OrderCategoricalLookup.TIMESLOT_OFFSET + 5 - 1] = 1 # Its after the fortieth minute
        self.assertEquals(timestamp_row, expected_timestamp_row)

    def test_january_11_first_minute(self):
        order_categorical_lookup = OrderCategoricalLookup()
        timestamp_row = order_categorical_lookup.get_timestamp_row('2016-01-11 00:00:59')
        expected_timestamp_row = [0] * OrderCategoricalLookup.TIMESTAMP_ROW_LENGTH
        expected_timestamp_row[0] = 1 # Its a Monday.
        expected_timestamp_row[7] = 0 # Its not a Holiday
        expected_timestamp_row[OrderCategoricalLookup.TIMESLOT_OFFSET + 1 - 1] = 1 # Its the first minute
        self.assertEquals(timestamp_row, expected_timestamp_row)

    def test_last_day_of_january_last_minute(self):
        order_categorical_lookup = OrderCategoricalLookup()
        timestamp_row = order_categorical_lookup.get_timestamp_row('2016-01-31 23:59:59')
        expected_timestamp_row = [0] * OrderCategoricalLookup.TIMESTAMP_ROW_LENGTH
        expected_timestamp_row[6] = 1 # Its a Sunday
        expected_timestamp_row[7] = 0 # Its not a Holiday
        expected_timestamp_row[OrderCategoricalLookup.TIMESTAMP_ROW_LENGTH - 1] = 1 # Its the last minute
        self.assertEquals(timestamp_row, expected_timestamp_row)

"""
Self test
"""
if __name__ == "__main__":
    unittest.main()
