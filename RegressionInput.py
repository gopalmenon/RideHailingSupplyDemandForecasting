from statistics import median
import FileIo
import logging
import OrderCategoricalLookup
import OrderKeyValue
import unittest

########################################################################################################################
#                                                                                                                      #
# Create input file for regression consisting of the categorical representation of start, destination districts and    #
# time slot number along with the associated number of orders and median price.                                        #
#                                                                                                                      #
########################################################################################################################


class RegressionInput(object):

    DRIVER_NOT_FOUND = "NULL"

    """
    Constructor
    """
    def __init__(self, data_set):

        self.data_set = data_set
        self.order_data = dict()
        self.input_to_regression_x_keys = list()
        self.input_to_regression_y_number_of_orders = list()
        self.input_to_regression_y_order_median_price = list()

        self.__summarize_order_data()
        self.order_categorical_lookup = OrderCategoricalLookup.OrderCategoricalLookup();
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

            logging.info("RegressionInput: Going through file " + file_name + " in " + self.data_set + " data")

            # Loop through the records and load the dictionary lookup
            for record in FileIo.get_text_file_contents(data_files_path + file_name):

                record_tokens = record.split(FileIo.TAB_CHARACTER)

                # If driver id is not present, skip the record
                order_driver_id = record_tokens[1]
                if order_driver_id == RegressionInput.DRIVER_NOT_FOUND:
                    continue

                # Create an order key to check if it already exists in summarized order data
                order_key = OrderKeyValue.OrderKey(record_tokens[3], record_tokens[4], record_tokens[6].strip())

                if order_key in self.order_data:
                    order_value = self.order_data[order_key]
                    order_value.append_order_price(record_tokens[5])
                    self.order_data[order_key] = order_value
                else:
                    self.order_data[order_key] = OrderKeyValue.OrderValue(record_tokens[5])


    """
    Use the summarized order data to generate inputs to the regression process
    """
    def __generate_input_to_regression(self):

        logging.info("RegressionInput: Building data for regression")

        # Loop through summarized data and generate inputs to the regression process
        for key, value in self.order_data.items():

            # Create a row for independent variables that will be the input for the prediction
            regression_key_values = list()
            regression_key_values\
                .append(self.order_categorical_lookup.get_district_hash_row(key.order_start_district))
            regression_key_values\
                .append(self.order_categorical_lookup.get_district_hash_row(key.order_destination_district))

            # Add categorical list for timestamp
            regression_key_values.append(OrderCategoricalLookup.OrderCategoricalLookup
                .get_timestamp_row_from_date_and_time_slot(key.order_date, key.order_time_slot))

            # Store the row
            self.input_to_regression_x_keys.append(regression_key_values)

            # Create two lists for median price and number of orders that will be the dependent variables
            self.input_to_regression_y_order_median_price.append(median(value.order_price))
            self.input_to_regression_y_number_of_orders.append(value.number_of_orders)

        self.order_data = None

    """
    Return inputs for regression
    """
    def get_regression_inputs(self):

        return self.input_to_regression_y_order_median_price, \
               self.input_to_regression_y_order_median_price, \
               self.input_to_regression_y_number_of_orders

########################################################################################################################
#                                                                                                                      #
# Unit testing class.                                                                                                  #
#                                                                                                                      #
########################################################################################################################


class TestRegressionInput(unittest.TestCase):

    def test_order_data_summarization(self):
        logging.getLogger().setLevel(logging.INFO)
        regression_input = RegressionInput("test")
        X, Y1, Y2 = regression_input.get_regression_inputs()
        print(str(len(X)) + " row generated for orders.")
        self.assertEquals(len(X), len(Y1), len(Y2))
