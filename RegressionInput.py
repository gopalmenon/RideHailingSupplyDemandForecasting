import FileIo
import logging
import numpy
import OrderCategoricalLookup
import OrderKeyValue
import src.POI.POI as POI
import TrafficLookup
import unittest
import WeatherLookup

########################################################################################################################
#                                                                                                                      #
# Create input file for regression consisting of the categorical representation of start, destination districts and    #
# time slot number along with the associated number of orders and median price.                                        #
#                                                                                                                      #
########################################################################################################################


class RegressionInput(object):

    TRAINING_DATA_POI_FILE_PATH = "data/season_1/training_data/poi_data/poi_data"
    TESTING_DATA_POI_FILE_PATH = "data/season_1/test_set_1/poi_data/poi_data"

    DRIVER_NOT_FOUND = "NULL"

    """
    Constructor
    """
    def __init__(self, data_set, order_categorical_lookup, poi_district_lookup):

        self.data_set = data_set
        self.order_data = dict()
        self.input_to_regression_x_keys = list()
        self.input_to_regression_y_number_of_orders = list()
        self.input_to_regression_y_order_median_price = list()

        self.poi_file_path = None
        if self.data_set == FileIo.TRAINING_DATA_SET:
            self.poi_file_path = RegressionInput.TRAINING_DATA_POI_FILE_PATH
        else:
            self.poi_file_path = RegressionInput.TESTING_DATA_POI_FILE_PATH
        self.poi_dictionary = POI.ReadPOI(self.poi_file_path).readFile()

        self.weather_lookup = WeatherLookup.WeatherLookup(data_set)

        self.traffic_lookup = TrafficLookup.TrafficLookup(data_set)

        self.poi_district_lookup = poi_district_lookup

        self.__summarize_order_data()
        self.order_categorical_lookup = order_categorical_lookup
        self.__generate_input_to_regression()

    """
    Add up the number of orders for a specific combination of date, time slot, start and end districts.
    Also maintain a list of prices that will be used to compute the median price.
    """
    def __summarize_order_data(self):

        if self.data_set == FileIo.TRAINING_DATA_SET:
            data_files_path = OrderCategoricalLookup.OrderCategoricalLookup.TRAINING_DATA_ORDER_FILES_PATH
            data_files_list = OrderCategoricalLookup.OrderCategoricalLookup.TRAINING_DATA_ORDER_FILES

        else:
            data_files_path = OrderCategoricalLookup.OrderCategoricalLookup.TESTING_DATA_ORDER_FILES_PATH
            data_files_list = OrderCategoricalLookup.OrderCategoricalLookup.TESTING_DATA_ORDER_FILES

        for file_name in data_files_list:

            logging.info("RegressionInput: Summarizing orders in file " + file_name + " in " + self.data_set + " data")

            # Loop through the records and load the dictionary lookup
            for record in FileIo.get_text_file_contents(data_files_path + file_name):

                record_tokens = record.split(FileIo.TAB_CHARACTER)

                # If driver id is not present, skip the record
                order_driver_id = record_tokens[1]
                if order_driver_id == RegressionInput.DRIVER_NOT_FOUND:
                    continue

                # Use the order for regression only if both start and end districts have POI and Traffic information
                # in both train and test environments
                if not self.traffic_lookup.district_has_traffic_info(district_hash=record_tokens[3]) or \
                   not self.traffic_lookup.district_has_traffic_info(district_hash=record_tokens[4]) or \
                   not self.poi_district_lookup.district_has_poi_info(district_hash=record_tokens[3]) or \
                   not self.poi_district_lookup.district_has_poi_info(district_hash=record_tokens[4]):
                    continue

                # Create an order key to check if it already exists in summarized order data
                order_key = OrderKeyValue.OrderKey(order_start_district=record_tokens[3],
                                                   order_destination_district=record_tokens[4],
                                                   order_timestamp=record_tokens[6].strip())

                if order_key in self.order_data:
                    order_value = self.order_data[order_key]
                    order_value.append_order_price(record_tokens[5])
                    self.order_data[order_key] = order_value
                else:
                    self.order_data[order_key] = OrderKeyValue.OrderValue(order_price=record_tokens[5])


    """
    Use the summarized order data to generate inputs to the regression process
    """
    def __generate_input_to_regression(self):

        logging.info("RegressionInput: Building data for regression")

        # Loop through summarized data and generate inputs to the regression process
        for key, value in self.order_data.items():

            # Create a row for independent variables that will be the input for the prediction.
            regression_key_values = list()

            # From district
            regression_key_values\
                .extend(self.order_categorical_lookup.get_district_hash_row(key.order_start_district))

            # From District POI (places of interest)
            #regression_key_values.extend(self.__get_poi_list(key.order_start_district))

            # From District traffic
            #regression_key_values\
            #    .extend(self.traffic_lookup
            #            .get_road_section_numbers_for_traffic_congestion_levels(key.order_start_district,
            #                                                                    key.order_timestamp))

            # To district
            regression_key_values\
                .extend(self.order_categorical_lookup.get_district_hash_row(key.order_destination_district))

            # To District POI (places of interest)
            #poi_list = self.__get_poi_list(key.order_destination_district)
            #regression_key_values.extend(self.__get_poi_list(key.order_destination_district))

            # To District traffic
            #regression_key_values \
            #    .extend(self.traffic_lookup
            #            .get_road_section_numbers_for_traffic_congestion_levels(key.order_destination_district,
            #                                                                    key.order_timestamp))

            # Add categorical list for timestamp
            regression_key_values.extend(OrderCategoricalLookup.OrderCategoricalLookup
                .get_timestamp_row_from_date_and_time_slot(key.order_date, key.order_time_slot))

            # Add categorical list for weather
            #regression_key_values.extend(self.weather_lookup.get_weather_snapshot(key.order_timestamp))

            # Store the row
            self.input_to_regression_x_keys.append(numpy.asarray(regression_key_values, dtype=numpy.float64))

            # Create two lists for median price and number of orders that will be the dependent variables
            number_of_orders, order_price = value.get_number_of_orders_and_median_price()
            self.input_to_regression_y_number_of_orders.append(number_of_orders)
            self.input_to_regression_y_order_median_price.append(order_price)

        self.order_data = None

    """
    Get POI categorical list given a district hash
    """
    def __get_poi_list(self, district_hash):

        if district_hash in self.poi_dictionary:
            return self.poi_dictionary[district_hash]
        else:
            raise ValueError('Could not find POI information for district ' + district_hash)

    """
    Return inputs for regression
    """
    def get_regression_inputs(self):

        logging.info("RegressionInput: Returning regression data with "
                     + str(len(self.input_to_regression_y_number_of_orders)) + " rows")

        return numpy.asarray(self.input_to_regression_x_keys), \
               numpy.asarray(self.input_to_regression_y_order_median_price, dtype=numpy.float64), \
               numpy.asarray(self.input_to_regression_y_number_of_orders, dtype=numpy.float64)

########################################################################################################################
#                                                                                                                      #
# Unit testing class.                                                                                                  #
#                                                                                                                      #
########################################################################################################################


class TestRegressionInput(unittest.TestCase):

    def test_order_data_summarization(self):
        logging.getLogger().setLevel(logging.INFO)
        regression_input = RegressionInput("test", OrderCategoricalLookup.OrderCategoricalLookup())
        order_start_end_districts_and_time, order_median_price, number_of_orders \
            = regression_input.get_regression_inputs()
        logging.info(str(len(order_start_end_districts_and_time)) + " row generated for orders.")
        self.assertEquals(len(order_start_end_districts_and_time), len(order_median_price), len(number_of_orders))
