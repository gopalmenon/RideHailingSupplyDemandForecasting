from sklearn import linear_model
import FileIo
import logging
import numpy
import OrderCategoricalLookup
import RegressionInput
import warnings

logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

order_categorical_lookup = OrderCategoricalLookup.OrderCategoricalLookup()

# Generate training inputs
training_regression_input = RegressionInput.RegressionInput(FileIo.TRAINING_DATA_SET, order_categorical_lookup)

# Generate regression model based on training data
training_order_start_end_districts_and_time, training_order_median_price, training_number_of_orders \
    = training_regression_input.get_regression_inputs()
reg = linear_model.SGDClassifier()
reg.fit (training_order_start_end_districts_and_time, training_number_of_orders)

# Get test data
testing_regression_input = RegressionInput.RegressionInput(FileIo.TEST_DATA_SET, order_categorical_lookup)
testing_order_start_end_districts_and_time, testing_order_median_price, testing_number_of_orders \
    = testing_regression_input.get_regression_inputs()
predicted_number_of_orders = reg.predict(testing_order_start_end_districts_and_time)

# Use mean squared error till accuracy metric is available
print("Mean squared error: %.2f", numpy.mean((predicted_number_of_orders - testing_number_of_orders) ** 2))