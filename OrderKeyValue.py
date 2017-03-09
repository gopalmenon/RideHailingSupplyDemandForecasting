import OrderCategoricalLookup as OCL

########################################################################################################################
#                                                                                                                      #
# Order Keys - Start District, Destination District, Date, Time Slot                                                   #
#                                                                                                                      #
########################################################################################################################


class OrderKey(object):

    """
    Constructor
    """
    def __init__(self, order_start_district, order_destination_district, order_timestamp):
        self.order_start_district = order_start_district
        self.order_destination_district = order_destination_district
        self.order_date = OCL.OrderCategoricalLookup.get_date_from_order_timestamp(order_timestamp)
        self.order_time_slot = OCL.OrderCategoricalLookup.get_time_slot_number_from_order_timestamp(order_timestamp)

    """
    Return hash for order keys
    """
    def __hash__(self):
        return hash((self.order_start_district, self.order_destination_district, self.order_date, self.order_time_slot))

    def __eq__(self, other):
        return (self.order_start_district, self.order_destination_district, self.order_date, self.order_time_slot) \
            == (other.order_start_district, other.order_destination_district, other.order_date, other.order_time_slot)

    def __ne__(self, other):
        return not(self == other)


########################################################################################################################
#                                                                                                                      #
# Order Value - Number of orders, array of order prices                                                                #
#                                                                                                                      #
########################################################################################################################

class OrderValue(object):

    """
    Constructor
    """
    def __init__(self, order_price):
        self.number_of_orders = 1
        self.order_price = [float(order_price)]

    """
    Append a price to the order summary. This will be appended to an array so that it can be used to compute a median
    price. Also increment the number of orders.
    """
    def append_order_price(self, order_price):
        self.number_of_orders += 1
        self.order_price.append(float(order_price))