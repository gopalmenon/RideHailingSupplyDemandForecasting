"""Return text file contents as a list"""
def get_text_file_contents(file_name):
    text_file = open(file_name, "r")
    text_file_contents = text_file.readlines()
    text_file.close()
    return text_file_contents