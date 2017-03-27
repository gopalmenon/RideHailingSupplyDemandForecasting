from math import exp
import numpy
import unittest

########################################################################################################################
#                                                                                                                      #
# Gaussian Kernel used for regression. This kernel will return the value for a query point based on the weighted       #
# average of the distance from the query point. The further away a data point is from a query point, the less it       #
# will contribute to the output.                                                                                       #
#                                                                                                                      #
########################################################################################################################


class GaussianKernel():

    GAUSSIAN_DISTRIBUTION_VARIANCE_VALUE = 0.01

    # This kernel function computes the weight to apply for data point based on its distance from the query point. The
    # part of the gaussian function before the exponent has been dropped as it will cancel out when the weighted
    # average is taken.
    @staticmethod
    def kernel(data_point_vector, query_point_vector):

        return exp(-1 * numpy.sum(numpy.square(data_point_vector - query_point_vector)) /
                   (2 * GaussianKernel.GAUSSIAN_DISTRIBUTION_VARIANCE_VALUE))

    # Return a prediction for the value at the query point by using the Gaussian Kernel
    @staticmethod
    def predict_query_point_value(training_data_points, training_data_values, query_point):

        prediction_weighted_value = 0.0
        prediction_weighted_sum = 0.0

        # Loop through each training data point and its value and compute a weighted average
        for data_point_index, data_point in enumerate(training_data_points):

            kernel_output = GaussianKernel.kernel(data_point_vector=training_data_points[data_point_index],
                                                  query_point_vector=query_point)

            prediction_weighted_value += kernel_output * training_data_values[data_point_index]
            prediction_weighted_sum += kernel_output

        # Return the prediction
        return prediction_weighted_value / prediction_weighted_sum

########################################################################################################################
#                                                                                                                      #
# Unit testing class.                                                                                                  #
#                                                                                                                      #
########################################################################################################################


class TestGaussianKernel(unittest.TestCase):

    def test_kernel_output(self):
        data_points = [[1,2,3],[4,5,6],[7,8,9]]
        data_values = [100,110,121]
        query_point = [2,4,7]
        prediction = GaussianKernel.predict_query_point_value(training_data_points=numpy.asarray(data_points),
                                                              training_data_values=numpy.asarray(data_values),
                                                              query_point=query_point)
        self.assertTrue(abs(prediction - 109.994472251)/prediction < 0.000001)

"""
Self test
"""
if __name__ == "__main__":
    unittest.main()
