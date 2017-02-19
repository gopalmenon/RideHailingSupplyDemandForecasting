import FileIo
import unittest
########################################################################################################################
#                                                                                                                      #
# DistrictLookup returns the district id given a district hash.                                                        #
#                                                                                                                      #
########################################################################################################################


class DistrictLookup(object):

    TRAINING_DATA_DISTRICT_FILE_PATH = "data/season_1/training_data/cluster_map/cluster_map"
    TESTING_DATA_DISTRICT_FILE_PATH = "data/season_1/test_set_1/cluster_map/cluster_map"

    """
    Constructor
    """
    def __init__(self, data_set):

        # Create storage to hold district data
        self.data_set = data_set
        self.district_data = dict()

        # Initialize file path based on which environment is being worked on
        data_files_path = None
        data_files_list = None
        if self.data_set == FileIo.TRAINING_DATA_SET:
            district_file_path = DistrictLookup.TRAINING_DATA_DISTRICT_FILE_PATH
        else:
            district_file_path = DistrictLookup.TESTING_DATA_DISTRICT_FILE_PATH

        # Fill lookup dictionary from the district file
        text_file_contents = FileIo.get_text_file_contents(district_file_path)

        # Loop through the records and load the dictionary lookup
        for record in text_file_contents:

            record_tokens = record.split(FileIo.TAB_CHARACTER)
            self.district_data[record_tokens[0]] = record_tokens[1].strip()

    """
    Return the district id corresponding to the district hash input parameter
    """
    def get_district_id(self, district_hash):

        # Look for a matching district hash in the dictionary
        try:
            return self.district_data[district_hash]
        except KeyError:
            return None


########################################################################################################################
#                                                                                                                      #
# Unit testing class.                                                                                                  #
#                                                                                                                      #
########################################################################################################################


class TestDistrictLookup(unittest.TestCase):

    def test_hash_exists(self):
        district_lookup = DistrictLookup("test")
        district_id = district_lookup.get_district_id("2350be163432e42270d2670cb3c02f80")
        self.assertEquals(district_id, '18')

    def test_hash_does_not_exist(self):
        district_lookup = DistrictLookup("test")
        district_id = district_lookup.get_district_id("2350be163432e424d53e1e42699cba6f")
        self.assertIsNone(district_id)

"""
Self test
"""
if __name__ == "__main__":
    unittest.main()
