import FileIo as FileIo
from datetime import datetime
import logging
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

    CHINESE_JANUARY_PUBLIC_HOLIDAYS =[datetime.strptime("2016-01-01", '%Y-%m-%d').date()]

    DAYS_IN_WEEK = 7
    HOLIDAY_SLOTS = 1
    TEN_MINUTE_SLOTS_IN_A_DAY = 144
    TIMESTAMP_ROW_LENGTH = DAYS_IN_WEEK + HOLIDAY_SLOTS + TEN_MINUTE_SLOTS_IN_A_DAY
    TIMESLOT_OFFSET = DAYS_IN_WEEK + HOLIDAY_SLOTS

    """
    Constructor
    """
    def __init__(self, poi_district_lookup):

        # Create storage for categorical data
        self.district_hashes = list()
        self.poi_district_lookup = poi_district_lookup

        unique_district_hashes = set()

        # Store all unique occurrences of categorical fields
        for data_set in [FileIo.TRAINING_DATA_SET, FileIo.TEST_DATA_SET]:

            logging.info("OrderCategoricalLookup: Going through data set " + data_set)

            if data_set == FileIo.TRAINING_DATA_SET:
                data_files_path = OrderCategoricalLookup.TRAINING_DATA_ORDER_FILES_PATH
                data_files_list = OrderCategoricalLookup.TRAINING_DATA_ORDER_FILES
            else:
                data_files_path = OrderCategoricalLookup.TESTING_DATA_ORDER_FILES_PATH
                data_files_list = OrderCategoricalLookup.TESTING_DATA_ORDER_FILES

            for file_name in data_files_list:

                logging.info("OrderCategoricalLookup: Going through file " + file_name + " in " + data_set
                             + " data for finding all districts")

                # Loop through the records and load the dictionary lookup
                for record in FileIo.get_text_file_contents(data_files_path + file_name):
                    record_tokens = record.split(FileIo.TAB_CHARACTER)
                    if self.poi_district_lookup.district_has_poi_info(record_tokens[3]):
                        unique_district_hashes.add(record_tokens[3])
                    if self.poi_district_lookup.district_has_poi_info(record_tokens[4]):
                        unique_district_hashes.add(record_tokens[4])

        # Store unique categorical field values
        self.district_hashes = list(unique_district_hashes)
        self.district_hashes.sort()

        logging.info("OrderCategoricalLookup: Found " + str(len(self.district_hashes)) + " districts")

    """
    Return a list corresponding to a district hash
    """
    def get_district_hash_row(self, district_hash):

        district_hash_row = [0] * len(self.district_hashes)
        district_hash_row[self.district_hashes.index(district_hash)] = 1
        return district_hash_row

    """
    Return a date given a timestamp
    """
    @staticmethod
    def get_date_from_order_timestamp(order_timestamp):

        return datetime.strptime(order_timestamp, FileIo.TIMESTAMP_FORMAT).date()

    """
    Return a time slot number given a datetime
    """
    @staticmethod
    def get_time_slot_number_from_order_datetime(order_datetime):

        # Each 24 hour day is divided up into 10 minute time slots. Find time slot that the order timestamp lies in.
        seconds_since_order_date_midnight = \
            (order_datetime - order_datetime.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
        ten_minute_steps_since_order_date_midnight = int(math.ceil(seconds_since_order_date_midnight / (60 * 10)))
        return ten_minute_steps_since_order_date_midnight


    """
    Return a time slot number given a timestamp
    """
    @staticmethod
    def get_time_slot_number_from_order_timestamp(order_timestamp):

        order_datetime = datetime.strptime(order_timestamp, FileIo.TIMESTAMP_FORMAT)
        return OrderCategoricalLookup.get_time_slot_number_from_order_datetime(order_datetime)

    """
    Return a list corresponding to a date and timeslot
    """
    @staticmethod
    def get_timestamp_row_from_date_and_time_slot(order_date, order_time_slot):

        # Create a timestamp row with the first 7 slots for day of the week, the 8th one for holiday yes/no, followed
        # by 144 slots for the 144 parts that a day is divided into.
        timestamp_row = [0] * OrderCategoricalLookup.TIMESTAMP_ROW_LENGTH

        # Get day of week corresponding to order date
        timestamp_row[order_date.weekday()] = 1

        # Check if order date is for a holiday
        if order_date in OrderCategoricalLookup.CHINESE_JANUARY_PUBLIC_HOLIDAYS:
            timestamp_row[7] = 1

        timestamp_row[OrderCategoricalLookup.TIMESLOT_OFFSET + order_time_slot - 1] = 1

        return timestamp_row

    """
    Return a list corresponding to a timestamp
    """
    @staticmethod
    def get_timestamp_row_from_timestamp(order_timestamp):

        order_datetime = datetime.strptime(order_timestamp, FileIo.TIMESTAMP_FORMAT)
        return OrderCategoricalLookup.get_timestamp_row_from_date_and_time_slot(order_datetime.date(),
                        OrderCategoricalLookup.get_time_slot_number_from_order_datetime(order_datetime))

########################################################################################################################
#                                                                                                                      #
# Unit testing class.                                                                                                  #
#                                                                                                                      #
########################################################################################################################


class TestOrderCategoricalLookup(unittest.TestCase):

    def test_fifth_timeslot_new_year_day(self):
        timestamp_row = OrderCategoricalLookup.get_timestamp_row_from_timestamp('2016-01-01 00:44:00')
        expected_timestamp_row = [0] * OrderCategoricalLookup.TIMESTAMP_ROW_LENGTH
        expected_timestamp_row[4] = 1 # Its a Friday. Zero offset corresponds to Monday.
        expected_timestamp_row[7] = 1 # Its a Holiday
        expected_timestamp_row[OrderCategoricalLookup.TIMESLOT_OFFSET + 5 - 1] = 1 # Its after the fortieth minute
        self.assertEquals(timestamp_row, expected_timestamp_row)

    def test_january_11_first_minute(self):
        timestamp_row = OrderCategoricalLookup.get_timestamp_row_from_timestamp('2016-01-11 00:00:59')
        expected_timestamp_row = [0] * OrderCategoricalLookup.TIMESTAMP_ROW_LENGTH
        expected_timestamp_row[0] = 1 # Its a Monday.
        expected_timestamp_row[7] = 0 # Its not a Holiday
        expected_timestamp_row[OrderCategoricalLookup.TIMESLOT_OFFSET + 1 - 1] = 1 # Its the first minute
        self.assertEquals(timestamp_row, expected_timestamp_row)

    def test_last_day_of_january_last_minute(self):
        timestamp_row = OrderCategoricalLookup.get_timestamp_row_from_timestamp('2016-01-31 23:59:59')
        expected_timestamp_row = [0] * OrderCategoricalLookup.TIMESTAMP_ROW_LENGTH
        expected_timestamp_row[6] = 1 # Its a Sunday
        expected_timestamp_row[7] = 0 # Its not a Holiday
        expected_timestamp_row[OrderCategoricalLookup.TIMESTAMP_ROW_LENGTH - 1] = 1 # Its the last minute
        self.assertEquals(timestamp_row, expected_timestamp_row)

    def test_date_from_timestamp(self):
        order_date = OrderCategoricalLookup.get_date_from_order_timestamp('2016-01-31 23:59:59')
        self.assertEquals(order_date, datetime.strptime('2016-01-31', '%Y-%m-%d').date())

"""
Self test
"""
if __name__ == "__main__":
    unittest.main()
