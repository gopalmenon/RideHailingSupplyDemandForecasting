from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import FileIo
import logging
import numpy
import OrderCategoricalLookup
import RegressionInput
import warnings


class RunRegressions(object):

    REGRESSION_TRAINING_INPUT_FILE_NAME = "RegressionTrainingInput.npz"
    REGRESSION_TESTING_INPUT_FILE_NAME = "RegressionTestingInput.npz"
    MAXIMUM_NUMBER_OF_JOBS = -1
    NUMBER_OF_CROSS_VALIDATION_FOLDS = 5

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

        logging.info("RunRegressions: Loaded " + str(len(self.training_number_of_orders)) + " training data rows")

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

        logging.info("RunRegressions: Loaded " + str(len(self.testing_number_of_orders)) + " testing data rows")


    """
    Run sgd regression
    """
    def __run_sgd_regression(self):

        losses = ["squared_loss"]
        penalties = ["none", "l2", "l1", "elasticnet"]
        initial_learning_rates =[0.1, 0.01, 0.001]
        learning_rates = ["constant", "optimal", "invscaling"]

        lowest_ride_prediction_error = float('inf')
        current_ride_prediction_error = 0.0

        best_loss = ""
        best_penalty = ""
        best_initial_learning_rate = 0.0
        best_learning_rate = ""

        # Find the best hyper-parameters
        for loss_counter in range(len(losses)):
            for penalty_counter in range(len(penalties)):
                for initial_learning_rate_counter in range(len(initial_learning_rates)):
                    for learning_rate_counter in range(len(learning_rates)):
                        current_ride_prediction_error = \
                        self.__run_sgd_regression_cross_validation(loss = losses[loss_counter],
                                                                penalty = penalties[penalty_counter],
                                                                initial_learning_rate =
                                                                  initial_learning_rates[initial_learning_rate_counter],
                                                                learning_rate = learning_rates[learning_rate_counter])

                        # Save values if better than previous best
                        if current_ride_prediction_error < lowest_ride_prediction_error:
                            lowest_ride_prediction_error = current_ride_prediction_error
                            best_loss = losses[loss_counter]
                            best_penalty = penalties[penalty_counter]
                            best_initial_learning_rate = initial_learning_rates[initial_learning_rate_counter]
                            best_learning_rate = learning_rates[learning_rate_counter]

        self.__run_sgd_regression_for_optimal_hyper_parameters(self,
                                                               loss = best_loss,
                                                               penalty = best_penalty,
                                                               initial_learning_rate = best_initial_learning_rate,
                                                               learning_rate = best_learning_rate)


    """
    Run sgd regression cross validation
    """
    def __run_sgd_regression_cross_validation(self, loss, penalty, initial_learning_rate, learning_rate):

        for cross_validation_fold in range(RunRegressions.NUMBER_OF_CROSS_VALIDATION_FOLDS):

            # Generate regression model based on training data
            logging.info("RunRegressions: " + str(RunRegressions.NUMBER_OF_CROSS_VALIDATION_FOLDS) +
                         " fold cross validating SGD Regressor with " +
                         str(len(self.training_number_of_orders)) + " rows, loss " + loss + ", penalty " + penalty +
                         ", initial learning rate " + str(initial_learning_rate) +
                         " and learning rate " + str(learning_rate))

            # Create the sgd regressor using the input parameters
            sgd_regressor = linear_model.SGDRegressor(loss = loss,
                                                      penalty = penalty,
                                                      eta0 = initial_learning_rate,
                                                      learning_rate = learning_rate)

            cross_validation_scores = cross_val_score(estimator = sgd_regressor,
                                                      X = self.training_order_start_end_districts_and_time,
                                                      y = self.training_order_median_price,
                                                      cv = RunRegressions.NUMBER_OF_CROSS_VALIDATION_FOLDS,
                                                      n_jobs = RunRegressions.MAXIMUM_NUMBER_OF_JOBS)

            return numpy.mean(cross_validation_scores)


    """
    Run sgd regression
    """
    def __run_sgd_regression_for_optimal_hyper_parameters(self, loss, penalty, initial_learning_rate, learning_rate):

        # Generate regression model based on training data
        logging.info("RunRegressions: Training SGD Regressor with " +
                     str(len(self.training_number_of_orders)) + " rows, loss " + loss + ", penalty " + penalty +
                     ", initial learning rate " + str(initial_learning_rate) + " and learning rate " +
                     str(learning_rate))

        # Create the sgd regressor using the input parameters
        sgd_regressor = linear_model.SGDRegressor(loss=loss,
                                                  penalty=penalty,
                                                  eta0=initial_learning_rate,
                                                  learning_rate=learning_rate)

        sgd_regressor.fit(self.training_order_start_end_districts_and_time, self.training_number_of_orders)

        # Predict the number of orders
        logging.info("RunRegressions: Predicting number of orders using SGD Regressor with "
                     + str(len(self.testing_number_of_orders)) + " rows")
        predicted_number_of_orders = sgd_regressor.predict(self.testing_order_start_end_districts_and_time)

        # Use mean squared error till accuracy metric is available
        print("Mean squared error in number of orders: " +
              str(numpy.mean((predicted_number_of_orders - self.testing_number_of_orders) ** 2)))


    """
    Reg all regressions
    """
    def run_regressions(self):

        self.__run_sgd_regression()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO,
                        # filename='regression_run.log', # log to this file
                        format='%(asctime)s %(message)s')  # include timestamp

    warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

    run_regressions = RunRegressions()
    run_regressions.run_regressions()
