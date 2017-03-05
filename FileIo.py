TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
TAB_CHARACTER = "\t"
TRAINING_DATA_SET = "train"
TEST_DATA_SET = "test"

"""Return text file contents as a generator"""


def get_text_file_contents(file_name):
    with open(file_name) as file_handle:
        for file_line in file_handle:
            yield file_line
