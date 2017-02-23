import unittest
import FileIo as FileIo


########################################################################################################################
#                                                                                                                      #
# This class stores all possible values for order fields and can return a categorical list for passenger id, driver id,#
# district and order time.                                                                                             #
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
Self test
"""
if __name__ == "__main__":
    order_categorical_lookup = OrderCategoricalLookup()
    driver_id_row = order_categorical_lookup.get_driver_id_row('12fdd642d752f20665ec521b00ae7071')
    print("Driver id row length: " + str(len(driver_id_row)))
    passenger_id_row = order_categorical_lookup.get_passenger_id_row('6a654832def9534c27c72f962e32c8f6')
    print("Passenger id row length: " + str(len(passenger_id_row)))
    district_hash_row = order_categorical_lookup.get_district_hash_row('f2c8c4bb99e6377d21de71275afd6cd2')
    print("District hash row length: " + str(len(district_hash_row)))
