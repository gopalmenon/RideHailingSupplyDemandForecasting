import FileIo
import unittest

########################################################################################################################
#                                                                                                                      #
# This class stores a list of all districts that have POI information in both train and test data                      #
#                                                                                                                      #
########################################################################################################################


class PoiDistrictLookup(object):

    TRAINING_DATA_POI_FILE_PATH = "data/season_1/training_data/poi_data/poi_data"
    TESTING_DATA_POI_FILE_PATH = "data/season_1/test_set_1/poi_data/poi_data"

    """
    Constructor
    """
    def __init__(self):

        self.districts_with_poi_info = None
        self.unique_training_districts_with_poi_info = set()
        self.unique_testing_districts_with_poi_info = set()

        # Loop through poi files in both train and test
        for data_set in [FileIo.TRAINING_DATA_SET, FileIo.TEST_DATA_SET]:

            if data_set == FileIo.TRAINING_DATA_SET:
                self.__load_districts(load_training_districts=True)
            else:
                self.__load_districts(load_training_districts=False)

        # Save list of districts that have poi info in both train and test environments
        self.districts_with_poi_info = list(self.unique_training_districts_with_poi_info
                                            .intersection(self.unique_testing_districts_with_poi_info))
        self.unique_training_districts_with_poi_info = None
        self.unique_testing_districts_with_poi_info = None

    """
    Load districts into set from poi file
    """
    def __load_districts(self, load_training_districts):

        if load_training_districts:
            file_name = PoiDistrictLookup.TRAINING_DATA_POI_FILE_PATH
        else:
            file_name = PoiDistrictLookup.TESTING_DATA_POI_FILE_PATH

        for file_line in FileIo.get_text_file_contents(file_name):

            file_line_fields = file_line.split(FileIo.TAB_CHARACTER)

            if load_training_districts:
                self.unique_training_districts_with_poi_info.add(file_line_fields[0])
            else:
                self.unique_testing_districts_with_poi_info.add(file_line_fields[0])

    """
    Return true if district has poi information in both train and test environments
    """
    def district_has_poi_info(self, district_hash):

        return district_hash in self.districts_with_poi_info

########################################################################################################################
#                                                                                                                      #
# Unit testing class.                                                                                                  #
#                                                                                                                      #
########################################################################################################################


class TestPoiDistrictLookup(unittest.TestCase):

    def test_districts_in_both_environments(self):
        poi_district_lookup = PoiDistrictLookup()
        self.assertTrue(poi_district_lookup.district_has_poi_info("87285a66236346350541b8815c5fae94"))

    def test_districts_without_poi_info(self):
        poi_district_lookup = PoiDistrictLookup()
        self.assertFalse(poi_district_lookup.district_has_poi_info("87285a66231236350541babc5c5fae94"))

"""
Self test
"""
if __name__ == "__main__":
    unittest.main()
