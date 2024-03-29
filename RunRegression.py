from numpy import linalg
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition.truncated_svd import TruncatedSVD
import FileIo
import GaussianKernel
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy
import OrderCategoricalLookup
import PoiDistrictLookup
import RegressionInput
import warnings


class RunRegression(object):

    REGRESSION_TRAINING_INPUT_FILE_NAME = "RegressionTrainingInput.npz"
    REGRESSION_TESTING_INPUT_FILE_NAME = "RegressionTestingInput.npz"
    DISTRICTS_VIZ_TEXT_FILE = "DistrictsViz.txt"
    TIMESLOTS_VIZ_TEXT_FILE = "TimeslotsViz.txt"
    RIDES_VIZ_TEXT_FILE = "RidesViz.txt"
    MAXIMUM_NUMBER_OF_JOBS = -1
    NUMBER_OF_CROSS_VALIDATION_FOLDS = 5
    ROWS_TO_USE_FOR_GAUSSIAN_KERNEL_REGRESSION = 15
    NUMBER_OF_TOP_EIGEN_VECTORS_TO_USE = 10
    NUMBER_OF_EIGEN_VECTORS_FOR_VISUALIZATION = 2
    DEGREE_OF_POLYNOMIAL_FOR_NON_LINEAR_REGRESSION = 4
    NUMBER_OF_DISTRICTS_WITH_TRAFFIC_AND_POI = 66

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

        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            self.__show_prediction_values_boxplots()

    """
    Show boxplots for number of rides and order price
    """
    def __show_prediction_values_boxplots(self):

        boxplots_data = [self.training_number_of_orders,
                         self.training_order_median_price,
                         self.testing_number_of_orders,
                         self.testing_order_median_price]

        # Show boxplots
        plt.figure()
        plt.boxplot(boxplots_data)
        plt.xticks([1, 2, 3, 4], ['Train Orders', 'Train Price', 'Test Orders', 'Test Price'])
        plt.show()

    """
    Run sgd regression
    """
    @staticmethod
    def run_sgd_regression(training_order_start_end_districts_and_time,
                           training_order_median_price,
                           training_number_of_orders,
                           testing_order_start_end_districts_and_time,
                           testing_order_median_price,
                           testing_number_of_orders):

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
                                                     len(training_order_start_end_districts_and_time) // \
                                                     RunRegression.NUMBER_OF_CROSS_VALIDATION_FOLDS

                                training_end_row = (fold_number + 1) * \
                                                   len(training_order_start_end_districts_and_time) // \
                                                    RunRegression.NUMBER_OF_CROSS_VALIDATION_FOLDS

                                logging.info("RunRegression: " + str(RunRegression.NUMBER_OF_CROSS_VALIDATION_FOLDS) +
                                             " fold cross validation training SGD Regressor for fold " +
                                             str(fold_number) + ", starting row " + str(training_start_row) +
                                             ", ending row " + str(training_end_row) + ", loss " + loss + ", penalty "
                                             + penalty + ", initial learning rate " + str(initial_learning_rate) +
                                             " and learning rate " + learning_rate)

                                # Train regression model
                                sgd_regressor\
                                   .partial_fit(X=training_order_start_end_districts_and_time[training_start_row :
                                                                                                   training_end_row],
                                                y=training_number_of_orders[training_start_row:training_end_row])

                            testing_start_row = testing_fold_number * \
                                                len(testing_order_start_end_districts_and_time) // \
                                                 RunRegression.NUMBER_OF_CROSS_VALIDATION_FOLDS

                            testing_end_row = (testing_fold_number + 1 )* \
                                                len(testing_order_start_end_districts_and_time) // \
                                                 RunRegression.NUMBER_OF_CROSS_VALIDATION_FOLDS

                            predicted_number_of_orders = sgd_regressor\
                                .predict(testing_order_start_end_districts_and_time[testing_start_row :
                                                                                         testing_end_row])

                            current_ride_prediction_error = numpy.mean((predicted_number_of_orders -
                                                                        testing_number_of_orders
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

        sgd_regressor.fit(X=training_order_start_end_districts_and_time,
                          y=training_number_of_orders)
        best_predicted_number_of_orders = sgd_regressor.predict(testing_order_start_end_districts_and_time)

        logging.info("RunRegression: Mean squared prediction error after cross validation is " +
                     str(numpy.mean((best_predicted_number_of_orders - testing_number_of_orders) ** 2)))

    """
    Check if mean prediction error is to high to qualify as the best so far
    """
    @staticmethod
    def __is_mean_prediction_error_too_high(cumulative_mean_prediction_error, best_prediction_error_so_far):

        return cumulative_mean_prediction_error / RunRegression.NUMBER_OF_CROSS_VALIDATION_FOLDS > \
               best_prediction_error_so_far

    """
    Run kernel ridge regression
    """
    def run_kernel_ridge_regression(self):

        logging.info("RunRegression: Starting kernel ridge regression")

        kernel_ridge_regressor = KernelRidge(alpha=1.0, kernel=GaussianKernel.GaussianKernel.get_kernel_value)
        kernel_ridge_regressor.fit(self.training_order_start_end_districts_and_time, self.training_number_of_orders)
        predicted_number_of_orders = kernel_ridge_regressor.predict(self.testing_order_start_end_districts_and_time)

        logging.info("RunRegression: Mean squared prediction error for kernel ridge regression is " +
                     str(numpy.mean((predicted_number_of_orders - self.testing_number_of_orders) ** 2)))


    """
    Run_data_visualization using first two eigen vectors
    """
    def run_data_visualization_using_eigen_vectors(self):

        training_data, testing_data = \
            self.__get_dimension_reduced_data_set(RunRegression.NUMBER_OF_EIGEN_VECTORS_FOR_VISUALIZATION)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Eigen Vector 1 Direction')
        ax.set_xticks([])
        ax.set_ylabel('Eigen Vector 2 Direction')
        ax.set_yticks([])
        ax.set_zlabel('Number of Rides')
        ax.set_title('Number of Rides for first 500K Training Points')
        ax.scatter(training_data[0:500000,0], training_data[0:500000,1], self.training_number_of_orders[0:500000])
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Eigen Vector 1 Direction')
        ax.set_xticks([])
        ax.set_ylabel('Eigen Vector 2 Direction')
        ax.set_yticks([])
        ax.set_zlabel('Median Ride Price')
        ax.set_title('Median Ride Price for first 500K Training Points')
        ax.scatter(training_data[0:500000,0], training_data[0:500000,1], self.training_order_median_price[0:500000])
        plt.show()

    """
    Run_data_visualization for start/end district and day of week
    """
    def run_data_visualization_for_start_loc_and_weekday(self, run_for_start_district):

        district_and_weekday_list = list()
        district_and_timeslot_list = list()

        for key_value in self.training_order_start_end_districts_and_time:

            if run_for_start_district:
                start_offset_for_district = 0
            else:
                start_offset_for_district = RunRegression.NUMBER_OF_DISTRICTS_WITH_TRAFFIC_AND_POI
            end_offset_for_district = start_offset_for_district + RunRegression.NUMBER_OF_DISTRICTS_WITH_TRAFFIC_AND_POI

            district_array = key_value[start_offset_for_district:end_offset_for_district]

            start_offset_for_day_of_week = 2 * RunRegression.NUMBER_OF_DISTRICTS_WITH_TRAFFIC_AND_POI
            end_offset_for_day_of_week = start_offset_for_day_of_week + \
                                         OrderCategoricalLookup.OrderCategoricalLookup.DAYS_IN_WEEK

            start_offset_for_timeslot = end_offset_for_day_of_week + 1
            timeslot_array = key_value[start_offset_for_timeslot : ]
            timeslot_index = list(timeslot_array).index(1)

            day_of_week_array = key_value[start_offset_for_day_of_week : end_offset_for_day_of_week]

            district_index = list(district_array).index(1)
            day_of_week_index = list(day_of_week_array).index(1)

            district_and_weekday_list.append([district_index, day_of_week_index])
            district_and_timeslot_list.append([district_index, timeslot_index])

        district_and_weekday_array = numpy.asarray(district_and_weekday_list)
        district_and_timeslot_array = numpy.asarray(district_and_timeslot_list)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if run_for_start_district:
            ax.set_xlabel('Start District')
        else:
            ax.set_xlabel('End District')
        ax.set_xticks([])
        ax.set_ylabel('Day of Week')
        ax.set_zlabel('Number of Rides')
        ax.set_title('Number of Rides for first 500K Training Points')
        ax.scatter(district_and_weekday_array[0:500000,0],
                   district_and_weekday_array[0:500000,1],
                   self.training_number_of_orders[0:500000])
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if run_for_start_district:
            ax.set_xlabel('Start District')
        else:
            ax.set_xlabel('End District')
        ax.set_xticks([])
        ax.set_ylabel('Time Slot')
        ax.set_zlabel('Number of Rides')
        ax.set_title('Number of Rides for first 500K Training Points')
        ax.scatter(district_and_timeslot_array[0:500000,0],
                   district_and_timeslot_array[0:500000,1],
                   self.training_number_of_orders[0:500000])
        plt.show()

        run_regression.__store_district_timeslot_visulaization_data(district_and_timeslot_array)


    """
    Store district, timeslots and rides data in a test file for display using Octave
    """
    def __store_district_timeslot_visulaization_data(self, district_and_timeslot_array):

        # Open files for write access and overwrite existing data
        try:
            with open(RunRegression.DISTRICTS_VIZ_TEXT_FILE, "w") as districts_file, \
                 open(RunRegression.TIMESLOTS_VIZ_TEXT_FILE, "w") as timeslots_file, \
                 open(RunRegression.RIDES_VIZ_TEXT_FILE, "w") as rides_file:

                # Loop through each record and write to text files
                for array_index, data_row in enumerate(district_and_timeslot_array):
                    districts_file.write(str(data_row[0])+'\n')
                    timeslots_file.write(str(data_row[1])+'\n')
                    rides_file.write(str(self.training_number_of_orders[array_index])+'\n')

        except IOError:
            logging.critical("IO Error while writing to vizulization files")

    """
    Run regression based on multidimensional scaling
    """
    def run_mds_regression(self):

        training_data, testing_data = \
            self.__get_dimension_reduced_data_set(RunRegression.NUMBER_OF_TOP_EIGEN_VECTORS_TO_USE)

        regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth = 1, random_state=0,
                                                 loss='ls').fit(training_data, self.training_number_of_orders)

        mean_square_prediction_error = \
            mean_squared_error(self.testing_number_of_orders, regressor.predict(testing_data))

        logging.info("RunRegression: Mean squared prediction error for gradient boosted regression is " +
                     str(mean_square_prediction_error))

        regressor = linear_model.Ridge (alpha = .5)\
            .fit(training_data, self.training_number_of_orders)

        mean_square_prediction_error = \
            mean_squared_error(self.testing_number_of_orders,
                               regressor.predict(testing_data))

        logging.info("RunRegression: Mean squared prediction error for Ridge regression is " +
                     str(mean_square_prediction_error))

        regressor = linear_model.Lasso (alpha = .5)\
            .fit(training_data, self.training_number_of_orders)

        mean_square_prediction_error = \
            mean_squared_error(self.testing_number_of_orders,
                               regressor.predict(testing_data))

        logging.info("RunRegression: Mean squared prediction error for Lasso regression is " +
                     str(mean_square_prediction_error))
    """
    Reduce dimensionality of data by finding projections onto the first k eigen vectors
    """
    def __get_dimension_reduced_data_set(self, number_of_dimensions_to_use):

        # Get first k eigen vectors
        first_k_eigen_vectors = self.__get_first_k_weighted_eigen_vectors(number_of_dimensions_to_use)

        # Return the training and testing data projections on the eigen vectors
        return numpy.dot(self.training_order_start_end_districts_and_time, first_k_eigen_vectors.T), \
               numpy.dot(self.testing_order_start_end_districts_and_time, first_k_eigen_vectors.T)

    """
    Return first k weighted eigen vectors
    """
    def __get_first_k_weighted_eigen_vectors(self, number_of_eigen_vectors_to_return):

        # Create a square matrix with number of test data rows preserved
        training_data_square_matrix = numpy.dot(self.training_order_start_end_districts_and_time.T,
                                                self.training_order_start_end_districts_and_time)

        logging.info("RunRegression: Square matrix shape " + str(training_data_square_matrix.shape))

        # Get Eigen values and eigen vectors
        training_data_eigen_values, training_data_eigen_vectors = linalg.eig(training_data_square_matrix)
        sorted_index = training_data_eigen_values.argsort()[::-1]
        sorted_training_data_eigen_values = training_data_eigen_values[sorted_index]
        sorted_training_data_eigen_vectors = training_data_eigen_vectors[:, sorted_index]

        logging.info("RunRegression: Found " + str(len(sorted_training_data_eigen_values)) + " eigen values.")
        logging.info("RunRegression: Eigen vectors have length " + str(len(sorted_training_data_eigen_vectors[0])))

        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            RunRegression.__show_eigen_values_trend(eigen_values=sorted_training_data_eigen_values)

        # Select only the first k eigen vectors and eigen values
        sorted_training_data_eigen_vectors = sorted_training_data_eigen_vectors[0:number_of_eigen_vectors_to_return, :]
        sorted_training_data_eigen_values = sorted_training_data_eigen_values[0:number_of_eigen_vectors_to_return]

        # Change the weight of the eigen vectors by the eigen values
        for eigen_value_index, eigen_value in enumerate(sorted_training_data_eigen_values):
            sorted_training_data_eigen_vectors[eigen_value_index] = \
                sorted_training_data_eigen_vectors[eigen_value_index] *\
                sorted_training_data_eigen_values[eigen_value_index]

        return sorted_training_data_eigen_vectors

    """
    Show Eigen values trend
    """
    @staticmethod
    def __show_eigen_values_trend(eigen_values):

        # Plot eigen values
        plt.plot(eigen_values)
        plt.ylabel('Eigen Value')
        plt.title('Sorted Eigen Values')
        plt.show()

    """
    Run kernel regression using a Guassian Kernel
    """
    def run_kernel_regression(self):

        predicted_rides = list()

        # Loop through training data and make predictions
        for testing_data_point_counter, testing_data_point in \
                enumerate(self.testing_order_start_end_districts_and_time
                          [:RunRegression.ROWS_TO_USE_FOR_GAUSSIAN_KERNEL_REGRESSION]):

            if testing_data_point_counter % 10 == 0:
                logging.info("RunRegression: Running prediction number " + str(testing_data_point_counter) +
                             " using Gaussian Regression")

            # Make a prediction and add it to list of predictions
            predicted_rides.append(GaussianKernel.GaussianKernel
                .predict_query_point_value(training_data_points=self.training_order_start_end_districts_and_time,
                                           training_data_values=self.training_number_of_orders,
                                           query_point=testing_data_point))

        logging.info("RunRegression: Mean squared prediction error using Gaussian Regression is " +
                     str(numpy.mean((numpy.asarray(predicted_rides) - self.testing_number_of_orders
                                                    [:RunRegression.ROWS_TO_USE_FOR_GAUSSIAN_KERNEL_REGRESSION]) ** 2)))

    def outliersPriceOrders(self,order,price):
        a = plt.boxplot([order,price])
        idx = set()
        idxSet = set(numpy.arange(len(self.training_order_start_end_districts_and_time)))
        for d in a['fliers']:
            print(len(d.get_ydata()))
            for point in d.get_ydata():
                pIdx = numpy.where((order == point)|(price==point))
                for rIdx in pIdx[0]:
                    idx.add(rIdx)
        logging.info("done with loop")
        idxKeep = list(idxSet.difference(idx))
        print(len(idxKeep))
        self.training_order_start_end_districts_and_time = self.training_order_start_end_districts_and_time[[idxKeep],:]
        self.training_number_of_orders = self.training_number_of_orders[[idxKeep]]
        self.training_order_median_price = self.training_order_median_price[[idxKeep]]
        self.training_order_start_end_districts_and_time = self.training_order_start_end_districts_and_time\
            .reshape(self.training_order_start_end_districts_and_time.shape[1:])
    
    def outliersSvdReduction(self):
        svd = TruncatedSVD(n_components=1)
        ordersSvd = svd.fit_transform(self.training_order_start_end_districts_and_time, self.training_number_of_orders)
        priceSvd = svd.fit_transform(self.training_order_start_end_districts_and_time, self.training_order_median_price)
        self.outliersPriceOrders(ordersSvd,priceSvd)

    """
    Run non-linear polynomial regression
    """
    def run_non_linear_polynomial_regression(self, polynomial_degree):

        logging.info("RunRegression: Convert features to polynomial of degree " +
                     str(polynomial_degree) +
                     " before running linear regression.")

        training_data, testing_data = \
            self.__get_dimension_reduced_data_set(RunRegression.NUMBER_OF_TOP_EIGEN_VECTORS_TO_USE)

        polynomial_features = PolynomialFeatures(degree=polynomial_degree)
        polynomial_features_training_data_keys = polynomial_features.fit_transform(training_data)
        polynomial_features_testing_data_keys = polynomial_features.fit_transform(testing_data)

        RunRegression\
            .run_sgd_regression(polynomial_features_training_data_keys,
                                self.training_order_median_price,
                                self.training_number_of_orders,
                                polynomial_features_testing_data_keys,
                                self.testing_order_median_price,
                                self.testing_number_of_orders)

if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG,
                        # filename='regression_run.log', # log to this file
                        format='%(asctime)s %(message)s')  # include timestamp

    warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

    run_regression = RunRegression()

    if logging.getLogger().getEffectiveLevel() == logging.INFO:
        run_regression.run_data_visualization_using_eigen_vectors()
        run_regression.run_data_visualization_for_start_loc_and_weekday(run_for_start_district=True)

    for polynomial_degree in range(2, RunRegression.DEGREE_OF_POLYNOMIAL_FOR_NON_LINEAR_REGRESSION+1):
        run_regression.run_non_linear_polynomial_regression(polynomial_degree)

    run_regression.outliersSvdReduction()
    RunRegression.run_sgd_regression(run_regression.training_order_start_end_districts_and_time,
                                     run_regression.training_order_median_price,
                                     run_regression.training_number_of_orders,
                                     run_regression.testing_order_start_end_districts_and_time,
                                     run_regression.testing_order_median_price,
                                     run_regression.testing_number_of_orders)
    run_regression.run_mds_regression()
    run_regression.run_kernel_regression()
