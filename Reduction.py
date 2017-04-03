from numpy import linalg
from sklearn import linear_model, svm
import FileIo
import GaussianKernel
import logging
import matplotlib.pyplot as plt
import numpy
import OrderCategoricalLookup
import PoiDistrictLookup
import RegressionInput
import warnings
from sklearn.decomposition.truncated_svd import TruncatedSVD
from sklearn.preprocessing.data import PolynomialFeatures

def sliceTransform(x, start, end):
    return x[start:end:1]

class RunRegression(object):

    REGRESSION_TRAINING_INPUT_FILE_NAME = "RegressionTrainingInput.npz"
    REGRESSION_TESTING_INPUT_FILE_NAME = "RegressionTestingInput.npz"
    MAXIMUM_NUMBER_OF_JOBS = -1
    NUMBER_OF_CROSS_VALIDATION_FOLDS = 5
    ROWS_TO_USE_FOR_GAUSSIAN_KERNEL_REGRESSION = 15
    DISTRICT_SIZE = 132
    TIME_SIZE = 152
    POI_SIZE = 352
    WEATHER_SIZE = 9
    TRAFFIC_SIZE = 8

    def __init__(self):
        
        self.components = 2
        self.svd = TruncatedSVD(n_components=self.components)
        self.reductCount = 0
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

                    self.dimensions = self.training_order_start_end_districts_and_time.shape[1]
                    self.initial = self.training_order_start_end_districts_and_time
                    logging.info("RunRegression: Loaded " + str(len(self.training_number_of_orders)) +
                                 " train data rows")
                else:

                    self.testing_order_start_end_districts_and_time, self.testing_order_median_price, \
                        self.testing_number_of_orders = saved_data['order_keys'], \
                                                        saved_data['order_value_price'], \
                                                        saved_data['order_value_number']

                    self.initialTesting = self.testing_order_start_end_districts_and_time
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
        
        coef = sgd_regressor.coef_
        print(coef)

        logging.info("RunRegression: Mean squared prediction error after cross validation is " +
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

        # Get Eigen values and eigen vectors
        training_data_eigen_values, training_data_eigen_vectors = linalg.eig(training_data_square_matrix)
        #print(training_data_eigen_values)
        #print(training_data_eigen_vectors)
        print(self.training_order_start_end_districts_and_time)
        sorted_index = training_data_eigen_values.argsort()[::-1]
        sorted_training_data_eigen_values = training_data_eigen_values[sorted_index]
        sorted_training_data_eigen_vectors = training_data_eigen_vectors[:, sorted_index]

        logging.info("RunRegression: Found " + str(len(sorted_training_data_eigen_values)) + " eigen values.")
        logging.info("RunRegression: Eigen vectors have length " + str(len(sorted_training_data_eigen_vectors[0])))

        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            RunRegression.__show_eigen_values_trend(eigen_values=sorted_training_data_eigen_values)

    """
    Show Eigen values trend
    """
    @staticmethod
    def __show_eigen_values_trend(self, eigen_values):

        # Plot eigen values
        plt.plot(eigen_values)
        plt.ylabel('Eigen Values')
        plt.title('Sorted Eigen Values')
        plt.show()
        
    def leastAngleRegression(self):
        lar = linear_model.Lars()
        lar.fit(self.training_order_start_end_districts_and_time, self.training_number_of_orders)
        predicted_number_of_orders = lar.predict(self.testing_order_start_end_districts_and_time)
        current_ride_prediction_error = numpy.mean((predicted_number_of_orders -
                                            self.testing_number_of_orders) ** 2)
        print(current_ride_prediction_error)
        print(lar.coef_)
        
    def orthogonalMatchingPursuit(self):
        omp = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=10)
        omp.fit(self.training_order_start_end_districts_and_time, self.training_number_of_orders)
        predicted_number_of_orders = omp.predict(self.testing_order_start_end_districts_and_time)
        current_ride_prediction_error = numpy.mean((predicted_number_of_orders -
                                            self.testing_number_of_orders) ** 2)
        print(current_ride_prediction_error)
        print(omp.coef_)
    
    def theilSenRegressor(self):
        tsr = linear_model.TheilSenRegressor()
        tsr.fit(self.training_order_start_end_districts_and_time, self.training_number_of_orders)
        predicted_number_of_orders = tsr.predict(self.testing_order_start_end_districts_and_time)
        current_ride_prediction_error = numpy.mean((predicted_number_of_orders -
                                            self.testing_number_of_orders) ** 2)
        print(current_ride_prediction_error)
        print(tsr.coef_)
        
    def polynomial(self):
        poly = PolynomialFeatures(degree=3)
        self.training_order_start_end_districts_and_time = poly.fit_transform(self.training_order_start_end_districts_and_time, self.training_number_of_orders)
        predict = poly.transform(self.testing_order_start_end_districts_and_time)
        
        clf = linear_model.LinearRegression()
        clf.fit(self.training_order_start_end_districts_and_time, self.training_number_of_orders)
        predicted_number_of_orders = clf.predict(predict)
        current_ride_prediction_error = numpy.mean((predicted_number_of_orders -
                                            self.testing_number_of_orders) ** 2)
        print(current_ride_prediction_error)
        print(clf.coef_)
     
    def svm(self):
        oneClass = svm.OneClassSVM()   
        logging.info("svm fit")
        oneClass.fit(self.training_order_start_end_districts_and_time, self.training_number_of_orders)
        logging.info("svm predict")
        predicted_number_of_orders = oneClass.predict(self.testing_order_start_end_districts_and_time)
        current_ride_prediction_error = numpy.mean((predicted_number_of_orders -
                                            self.testing_number_of_orders) ** 2)
        print(current_ride_prediction_error)
        print(oneClass.coef_)
    
    def districtReduction(self,keyType,key):
        y = key
        districts = numpy.apply_along_axis(sliceTransform, 1, y,0,self.DISTRICT_SIZE)
        if keyType == "training":          
            districtRed = self.svd.fit_transform(districts,self.training_number_of_orders)
        else:            
            districtRed = self.svd.transform(districts)
        nonDistrict = numpy.apply_along_axis(sliceTransform, 1, y,self.DISTRICT_SIZE,self.dimensions)
        keyWithDist = numpy.append(districtRed, nonDistrict, axis=1)
        return keyWithDist
        
    def timeReduction(self,keyType,key):
        y = key
        time = numpy.apply_along_axis(sliceTransform, 1, y,self.components,self.TIME_SIZE+self.components)
        if keyType == "training":
            timeRed = self.svd.fit_transform(time,self.training_number_of_orders)
        else:
            timeRed = self.svd.transform(time)
        befTime = numpy.apply_along_axis(sliceTransform, 1, y,0,self.components)
        aftTime = numpy.apply_along_axis(sliceTransform, 1, y,self.TIME_SIZE+self.components,self.dimensions)
        keyWithTime = numpy.append(befTime, timeRed, axis=1)
        keyWithTime = numpy.append(keyWithTime,aftTime,axis=1)
        return keyWithTime
        
    def POIReduction(self,keyType,key):
        y = key
        poi = numpy.apply_along_axis(sliceTransform, 1, y,self.components*2,self.POI_SIZE+self.components*2)
        if keyType == "training":
            poiRed = self.svd.fit_transform(poi,self.training_number_of_orders)
        else:
            poiRed = self.svd.transform(poi)
        befPoi = numpy.apply_along_axis(sliceTransform, 1, y,0,self.components*2)
        aftPoi = numpy.apply_along_axis(sliceTransform, 1, y,self.POI_SIZE+self.components*2,self.dimensions)
        keyWithPoi = numpy.append(befPoi, poiRed, axis=1)
        keyWithPoi = numpy.append(keyWithPoi, aftPoi, axis=1)
        return keyWithPoi
        
    def WeatherReduction(self,keyType,key):
        y = key
        weather = numpy.apply_along_axis(sliceTransform, 1, y,self.components*3,self.WEATHER_SIZE+self.components*3)
        if keyType == "training":
            weatherRed = self.svd.fit_transform(weather,self.training_number_of_orders)
        else:
            weatherRed = self.svd.transform(weather)
        befWeather = numpy.apply_along_axis(sliceTransform, 1, y,0,self.components*3)
        aftWeather = numpy.apply_along_axis(sliceTransform, 1, y,self.WEATHER_SIZE+self.components*3,self.dimensions)
        keyWithWeather = numpy.append(befWeather, weatherRed, axis=1)
        keyWithWeather = numpy.append(keyWithWeather, aftWeather,axis=1)
        return keyWithWeather
        
    def TrafficReduction(self,keyType,key):
        y = key
        traffic = numpy.apply_along_axis(sliceTransform, 1, y,self.components*4,self.TRAFFIC_SIZE+self.components*4)
        if keyType == "training":
            trafficRed = self.svd.fit_transform(traffic,self.training_number_of_orders)
            if self.reductCount == 0:
                self.boxPlot(trafficRed)
                self.reductCount = 1
        else:
            trafficRed = self.svd.transform(traffic)
        befTraffic = numpy.apply_along_axis(sliceTransform, 1, y,0,self.components*4)
        keyWithTraffic = numpy.append(befTraffic, trafficRed, axis=1)
        return keyWithTraffic

    def wholeReductionTraining(self):
        y = self.training_order_start_end_districts_and_time
        b = self.svd.fit_transform(y,self.training_number_of_orders)
        if self.reductCount <2:
            self.boxPlot(b)
        self.reductCount +=1
        self.training_order_start_end_districts_and_time = b
        
    def wholeReductionTesting(self):
        y = self.testing_order_start_end_districts_and_time
        b = self.svd.transform(y)
        self.testing_order_start_end_districts_and_time = b
    
    def reduction(self):
        self.training_order_start_end_districts_and_time = self.initial
        self.dimensions = self.training_order_start_end_districts_and_time.shape[1]
        self.testing_order_start_end_districts_and_time = self.initialTesting
        
        logging.info("RunRegression: Reducing Districts")
        self.training_order_start_end_districts_and_time = run_regression.districtReduction('training', self.training_order_start_end_districts_and_time)
        self.testing_order_start_end_districts_and_time = run_regression.districtReduction('testing',self.testing_order_start_end_districts_and_time)
        x = self.testing_order_start_end_districts_and_time[:,0:1]
        y = self.testing_order_start_end_districts_and_time[:,1:2]
        self.dimensions = self.training_order_start_end_districts_and_time.shape[1]
        print(self.dimensions)
        
        logging.info("RunRegression: Reducing Time")
        self.training_order_start_end_districts_and_time = run_regression.timeReduction('training', self.training_order_start_end_districts_and_time)
        self.testing_order_start_end_districts_and_time = run_regression.timeReduction('testing',self.testing_order_start_end_districts_and_time)
        x = self.training_order_start_end_districts_and_time[:,2:3]
        y = self.training_order_start_end_districts_and_time[:,3:4]
        self.dimensions = self.training_order_start_end_districts_and_time.shape[1]
        #plt.scatter(x,y)
        #plt.show()
        logging.info("RunRegression: Reducing POI")
        self.training_order_start_end_districts_and_time = run_regression.POIReduction('training', self.training_order_start_end_districts_and_time)
        self.testing_order_start_end_districts_and_time = run_regression.POIReduction('testing',self.testing_order_start_end_districts_and_time)
        x = self.training_order_start_end_districts_and_time[:,4:5]
        y = self.training_order_start_end_districts_and_time[:,5:6]
        self.dimensions = self.training_order_start_end_districts_and_time.shape[1]
        #plt.scatter(x,y)
        #plt.show()
        logging.info("RunRegression: Reducing Weather")
        self.training_order_start_end_districts_and_time = run_regression.WeatherReduction('training', self.training_order_start_end_districts_and_time)
        self.testing_order_start_end_districts_and_time = run_regression.WeatherReduction('testing',self.testing_order_start_end_districts_and_time)
        x = self.training_order_start_end_districts_and_time[:,6:7]
        y = self.training_order_start_end_districts_and_time[:,7:8]
        self.dimensions = self.training_order_start_end_districts_and_time.shape[1]
        #plt.scatter(x,y)
        #plt.show()
        logging.info("RunRegression: Reducing Traffic")
        self.training_order_start_end_districts_and_time = run_regression.TrafficReduction('training', self.training_order_start_end_districts_and_time)
        self.testing_order_start_end_districts_and_time = run_regression.TrafficReduction('testing',self.testing_order_start_end_districts_and_time)
        x = self.training_order_start_end_districts_and_time[:,8:9]
        y = self.training_order_start_end_districts_and_time[:,9:10]
        self.dimensions = self.training_order_start_end_districts_and_time.shape[1]
        print(self.initial.shape)

    def boxPlot(self, arrayBox):
        a = plt.boxplot(arrayBox)
        plt.show()
        idx = set()
        idxSet = set(numpy.arange(len(self.training_order_start_end_districts_and_time)))
        for d in a['fliers']:
            print(len(d.get_ydata()))
            for point in d.get_ydata():
                pIdx = numpy.where(arrayBox == point)
                for rIdx in pIdx[0]:
                    idx.add(rIdx)
        logging.info("done with loop")
        idxKeep = list(idxSet.difference(idx))
        self.initial = self.initial[[idxKeep],:]
        self.training_number_of_orders = self.training_number_of_orders[[idxKeep]]
        self.initial = self.initial.reshape(self.initial.shape[1:])

        #plt.scatter(x,y)
        #plt.show()
        
        

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO,
                        # filename='regression_run.log', # log to this file
                        format='%(asctime)s %(message)s')  # include timestamp

    warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

    run_regression = RunRegression()
    #run_regression.reduction()
    #run_regression.components = 2
    #run_regression.svd = TruncatedSVD(n_components=run_regression.components)
    #run_regression.reduction()
    '''
    run_regression.wholeReductionTraining()
    run_regression.training_order_start_end_districts_and_time = run_regression.initial
    run_regression.components = 2
    run_regression.svd = TruncatedSVD(n_components=run_regression.components)
    run_regression.wholeReductionTraining()
    run_regression.training_order_start_end_districts_and_time = run_regression.initial
    run_regression.wholeReductionTraining()
    run_regression.wholeReductionTesting()
    '''
    #run_regression.run_sgd_regression()
    #run_regression.run_mds_regression()
    #run_regression.leastAngleRegression()
    #run_regression.orthogonalMatchingPursuit()
    #run_regression.theilSenRegressor()
    #run_regression.polynomial()
    run_regression.reductCount = 3
    logging.info("reduction training")
    run_regression.wholeReductionTraining()
    logging.info("reduction testing")
    run_regression.wholeReductionTesting()
    logging.info("svm")
    run_regression.svm()
