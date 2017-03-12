from sklearn import linear_model
import FileIo
import logging
import numpy
import OrderCategoricalLookup
import RegressionInput
import warnings


class RunRegressions(object):

    REGRESSION_TRAINING_INPUT_FILE_NAME = "RegressionTrainingInput.npz"
    REGRESSION_TESTING_INPUT_FILE_NAME = "RegressionTestingInput.npz"

    def __init__(self):

        self.order_categorical_lookup = None

        # Check and see if training data has already been saved
        try:
            saved_training_data = numpy.load(RunRegressions.REGRESSION_TRAINING_INPUT_FILE_NAME, mmap_mode='r')

        # If training data is not found, load it
        except IOError:

            logging.info("RunRegressions: Generating training data")
            self.order_categorical_lookup = OrderCategoricalLookup.OrderCategoricalLookup()

            # Generate training inputs
            training_regression_input = RegressionInput.RegressionInput(FileIo.TRAINING_DATA_SET,
                                                                        self.order_categorical_lookup)
            self.training_order_start_end_districts_and_time, \
            self.training_order_median_price, \
            self.training_number_of_orders \
                = training_regression_input.get_regression_inputs()

            # Save the training data for next time
            numpy.savez(RunRegressions.REGRESSION_TRAINING_INPUT_FILE_NAME,
                        order_keys = self.training_order_start_end_districts_and_time,
                        order_value_price = self.training_order_median_price,
                        order_value_number = self.training_number_of_orders)

        # If the saved training data is found, load it
        else:

            logging.info("RunRegressions: Loading training data")
            self.training_order_start_end_districts_and_time, \
            self.training_order_median_price, \
            self.training_number_of_orders \
                = saved_training_data['order_keys'], \
                  saved_training_data['order_value_price'], \
                  saved_training_data['order_value_number']

        # Check and see if testing data has already been saved
        try:
            saved_testing_data = numpy.load(RunRegressions.REGRESSION_TESTING_INPUT_FILE_NAME, mmap_mode='r')

        # If testing data is not found, load it
        except IOError:

            logging.info("RunRegressions: Generating testing data")

            if self.order_categorical_lookup is None:
                self.order_categorical_lookup = OrderCategoricalLookup.OrderCategoricalLookup()

            # Generate testing inputs
            testing_regression_input = RegressionInput.RegressionInput(FileIo.TEST_DATA_SET,
                                                                        self.order_categorical_lookup)
            self.testing_order_start_end_districts_and_time, \
            self.testing_order_median_price, \
            self.testing_number_of_orders \
                = testing_regression_input.get_regression_inputs()

            # Save the testing data for next time
            numpy.savez(RunRegressions.REGRESSION_TESTING_INPUT_FILE_NAME,
                        order_keys = self.testing_order_start_end_districts_and_time,
                        order_value_price = self.testing_order_median_price,
                        order_value_number = self.testing_number_of_orders)

        # If the saved testing data is found, load it
        else:

            logging.info("RunRegressions: Loading testing data")
            self.testing_order_start_end_districts_and_time, \
            self.testing_order_median_price, \
            self.testing_number_of_orders \
                = saved_testing_data['order_keys'], \
                  saved_testing_data['order_value_price'], \
                  saved_testing_data['order_value_number']

    """
    Run the regression
    """
    def run_regressions(self):

        # Generate regression model based on training data
        sgd_regressor = linear_model.SGDRegressor()
        sgd_regressor.fit (self.training_order_start_end_districts_and_time, self.training_number_of_orders)

        # Predict the number of orders
        predicted_number_of_orders = sgd_regressor.predict(self.testing_order_start_end_districts_and_time)

        # Use mean squared error till accuracy metric is available
        print("Mean squared error: " +
              str(numpy.mean((predicted_number_of_orders - self.testing_number_of_orders) ** 2)))

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO,
                        # filename='regression_run.log', # log to this file
                        format='%(asctime)s %(message)s')  # include timestamp

    warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

    run_regressions = RunRegressions()
    run_regressions.run_regressions()
