from sklearn import linear_model
import FileIo
import logging
import numpy
import RegressionInput
import warnings

logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

regression_input = RegressionInput.RegressionInput(FileIo.TEST_DATA_SET)

reg = linear_model.LinearRegression()
order_start_end_districts_and_time, order_median_price, number_of_orders = regression_input.get_regression_inputs()
reg.fit (order_start_end_districts_and_time, number_of_orders)
print(reg)