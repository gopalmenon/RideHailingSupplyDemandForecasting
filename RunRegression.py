from numpy import linalg
from sklearn import linear_model
import FileIo
import GaussianKernel
import logging
import matplotlib.pyplot as plt
import numpy
import OrderCategoricalLookup
import PoiDistrictLookup
import RegressionInput
import warnings


class RunRegression(object):

    REGRESSION_TRAINING_INPUT_FILE_NAME = "RegressionTrainingInput.npz"
    REGRESSION_TESTING_INPUT_FILE_NAME = "RegressionTestingInput.npz"
    MAXIMUM_NUMBER_OF_JOBS = -1
    NUMBER_OF_CROSS_VALIDATION_FOLDS = 5

    def __init__(self):

        for file_name, data_set in [(RunRegression.REGRESSION_TRAINING_INPUT_FILE_NAME, FileIo.TRAINING_DATA_SET),
                                    (RunRegression.REGRESSION_TESTING_INPUT_FILE_NAME, FileIo.TEST_DATA_SET)]:

            # Check and see if the data has already been saved
            try:

                logging.info("RunRegression: Trying to load " + data_set + " data")

                saved_data = numpy.load(file_name, mmap_mode='r')

            # If the data is not found, load it
            except IOError:

                logging.info("RunRegression: Saved data not found. Generating " + data_set + " data")

                # Generate inputs
                poi_district_lookup = PoiDistrictLookup.PoiDistrictLookup()
                order_categorical_lookup = OrderCategoricalLookup.OrderCategoricalLookup(poi_district_lookup)
                regression_input = RegressionInput.RegressionInput(data_set,
                                                                   order_categorical_lookup,
                                                                   poi_district_lookup)

                if data_set == FileIo.TRAINING_DATA_SET:

                    self.training_order_start_end_districts_and_time, self.training_order_median_price, \
                        self.training_number_of_orders = regression_input.get_regression_inputs()

                    # Save the data for next time
                    numpy.savez(file_name,
                                order_keys=self.training_order_start_end_districts_and_time,
                                order_value_price=self.training_order_median_price,
                                order_value_number=self.training_number_of_orders)

                else:

                    self.testing_order_start_end_districts_and_time, self.testing_order_median_price, \
                        self.testing_number_of_orders  = regression_input.get_regression_inputs()

                    # Save the data for next time
                    numpy.savez(file_name,
                                order_keys=self.testing_order_start_end_districts_and_time,
                                order_value_price=self.testing_order_median_price,
                                order_value_number=self.testing_number_of_orders)

            # If the saved data is found, load it
            else:

                logging.info("RunRegression: Loading " + data_set + " data")

                if data_set == FileIo.TRAINING_DATA_SET:

                    self.training_order_start_end_districts_and_time, self.training_order_median_price, \
                        self.training_number_of_orders = saved_data['order_keys'], \
                                                         saved_data['order_value_price'], \
                                                         saved_data['order_value_number']

                    logging.info("RunRegression: Loaded " + str(len(self.training_number_of_orders)) +
                                 " train data rows")
                else:

                    self.testing_order_start_end_districts_and_time, self.testing_order_median_price, \
                        self.testing_number_of_orders = saved_data['order_keys'], \
                                                        saved_data['order_value_price'], \
                                                        saved_data['order_value_number']

                    logging.info("RunRegression: Loaded " + str(len(self.testing_number_of_orders)) +
                                 " test data rows")

    """
    Run sgd regression
    """
    def run_sgd_regression(self):

        losses = ["squared_loss"]
        penalties = ["none", "l2", "l1", "elasticnet"]
        initial_learning_rates =[0.1, 0.01, 0.001]
        learning_rates = ["constant", "optimal", "invscaling"]

        lowest_ride_prediction_error = float('inf')

        best_loss = ""
        best_penalty = ""
        best_initial_learning_rate = 0.0
        best_learning_rate = ""

        # Find the best hyper-parameters
        for loss in losses:
            for penalty in penalties:
                for initial_learning_rate in initial_learning_rates:
                    for learning_rate in learning_rates:

                        mean_ride_prediction_error = 0.0

                        # Do k-fold cross-validation using mini-batch training.
                        for testing_fold_number in range(RunRegression.NUMBER_OF_CROSS_VALIDATION_FOLDS):

                            # Create the sgd regressor using the input parameters
                            sgd_regressor = linear_model.SGDRegressor(loss=loss,
                                                                      penalty=penalty,
                                                                      eta0=initial_learning_rate,
                                                                      learning_rate=learning_rate)

                            # Run mini batch training for the fold if its not the training fold
                            for fold_number in range(RunRegression.NUMBER_OF_CROSS_VALIDATION_FOLDS):

                                if fold_number == testing_fold_number:
                                    continue

                                training_start_row = fold_number * \
                                                     len(self.training_order_start_end_districts_and_time) // \
                                                     RunRegression.NUMBER_OF_CROSS_VALIDATION_FOLDS

                                training_end_row = (fold_number + 1) * \
                                                   len(self.training_order_start_end_districts_and_time) // \
                                                    RunRegression.NUMBER_OF_CROSS_VALIDATION_FOLDS

                                logging.info("RunRegression: " + str(RunRegression.NUMBER_OF_CROSS_VALIDATION_FOLDS) +
                                             " fold cross validation training SGD Regressor for fold " +
                                             str(fold_number) + ", starting row " + str(training_start_row) +
                                             ", ending row " + str(training_end_row) + ", loss " + loss + ", penalty "
                                             + penalty + ", initial learning rate " + str(initial_learning_rate) +
                                             " and learning rate " + learning_rate)

                                # Train regression model
                                sgd_regressor\
                                   .partial_fit(X=self.training_order_start_end_districts_and_time[training_start_row :
                                                                                                   training_end_row],
                                                y=self.training_number_of_orders[training_start_row:training_end_row])

                            testing_start_row = testing_fold_number * \
                                                len(self.testing_order_start_end_districts_and_time) // \
                                                 RunRegression.NUMBER_OF_CROSS_VALIDATION_FOLDS

                            testing_end_row = (testing_fold_number + 1 )* \
                                                len(self.testing_order_start_end_districts_and_time) // \
                                                 RunRegression.NUMBER_OF_CROSS_VALIDATION_FOLDS

                            predicted_number_of_orders = sgd_regressor\
                                .predict(self.testing_order_start_end_districts_and_time[testing_start_row :
                                                                                         testing_end_row])

                            current_ride_prediction_error = numpy.mean((predicted_number_of_orders -
                                                                        self.testing_number_of_orders
                                                                        [testing_start_row : testing_end_row]) ** 2)

                            logging.info("RunRegression: Prediction error for fold " + str(testing_fold_number) +
                                         " is " + str(current_ride_prediction_error))

                            mean_ride_prediction_error += current_ride_prediction_error

                            if RunRegression.__is_mean_prediction_error_too_high(mean_ride_prediction_error,
                                                                                 lowest_ride_prediction_error):
                                logging.info("RunRegression: Mean prediction error of " +
                                             str(mean_ride_prediction_error) + "is too high compared to best so far " +
                                             str(lowest_ride_prediction_error) + ". Ending current cross validation.")
                                break

                        else:

                            mean_ride_prediction_error /= RunRegression.NUMBER_OF_CROSS_VALIDATION_FOLDS

                            logging.info("RunRegression: Mean prediction error is " + str(mean_ride_prediction_error))

                            # Save values if better than previous best
                            if mean_ride_prediction_error < lowest_ride_prediction_error:

                                logging.info("RunRegression: mean error of " + str(mean_ride_prediction_error) +
                                             " is the best so far. Saving loss " + loss + ", penalty " + penalty +
                                             ", initial learning rate " + str(initial_learning_rate) +
                                             " and learning rate " + learning_rate)

                                lowest_ride_prediction_error = mean_ride_prediction_error
                                best_loss = loss
                                best_penalty = penalty
                                best_initial_learning_rate = initial_learning_rate
                                best_learning_rate = learning_rate

        logging.info("RunRegression: Running regression with best values so far: loss " + best_loss + ", penalty " +
                     best_penalty + ", initial learning rate " + str(best_initial_learning_rate) +
                     " and learning rate " + best_learning_rate)

        sgd_regressor = linear_model.SGDRegressor(loss=best_loss,
                                                  penalty=best_penalty,
                                                  eta0=best_initial_learning_rate,
                                                  learning_rate=best_learning_rate)

        sgd_regressor.fit(X=self.training_order_start_end_districts_and_time,
                          y=self.training_number_of_orders)
        best_predicted_number_of_orders = sgd_regressor.predict(self.testing_order_start_end_districts_and_time)

        logging.info("RunRegression: Mean prediction error after cross validation is " +
                     str(numpy.mean((best_predicted_number_of_orders - self.testing_number_of_orders) ** 2)))

    """
    Check if mean prediction error is to high to qualify as the best so far
    """
    @staticmethod
    def __is_mean_prediction_error_too_high(cumulative_mean_prediction_error, best_prediction_error_so_far):

        return cumulative_mean_prediction_error / RunRegression.NUMBER_OF_CROSS_VALIDATION_FOLDS > \
               best_prediction_error_so_far


    """
    Run regression based on multidimensional scaling
    """
    def run_mds_regression(self):

        # Create a square matrix with number of test data rows preserved
        training_data_square_matrix = numpy.dot(self.training_order_start_end_districts_and_time.T,
                                                self.training_order_start_end_districts_and_time)

        logging.info("RunRegression: Square matrix shape " + str(training_data_square_matrix.shape))

        # Get eigen values and eigen vectors
        training_data_eigen_values, training_data_eigen_vectors = linalg.eig(training_data_square_matrix)
        sorted_index = training_data_eigen_values.argsort()[::-1]
        sorted_training_data_eigen_values = training_data_eigen_values[sorted_index]
        sorted_training_data_eigen_vectors = training_data_eigen_vectors[:, sorted_index]

        logging.info("RunRegression: Found " + str(len(sorted_training_data_eigen_values)) + " eigen values.")
        logging.info("RunRegression: Eigen vectors have length " + str(len(sorted_training_data_eigen_vectors[0])))

        # Plot eigen values
        plt.plot(sorted_training_data_eigen_values)
        plt.ylabel('Eigen Values')
        plt.title('Sorted Eigen Values')
        plt.show()


    """
    Run kernel regression using a Guassian Kernel
    """
    def run_kernel_regression(self):

        predicted_rides = list()

        # Loop through training data and make predictions
        for testing_data_point_index, testing_data_point in enumerate(self.testing_order_start_end_districts_and_time):

            if testing_data_point_index % 10 == 0:
                logging.info("RunRegression: Running prediction number " + str(testing_data_point_index) +
                             " using Gaussian Regression")

            # Make a prediction and add it to list of predictions
            predicted_rides.append(GaussianKernel.GaussianKernel
                .predict_query_point_value(training_data_points=self.training_order_start_end_districts_and_time,
                                           training_data_values=self.training_number_of_orders,
                                           query_point=testing_data_point))

        logging.info("RunRegression: Mean prediction error using Gaussian Regression is " +
                     str(numpy.mean((numpy.asarray(predicted_rides) - self.testing_number_of_orders) ** 2)))

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO,
                        # filename='regression_run.log', # log to this file
                        format='%(asctime)s %(message)s')  # include timestamp

    warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

    run_regression = RunRegression()
    #run_regression.run_sgd_regression()
    #run_regression.run_mds_regression()
    run_regression.run_kernel_regression()
