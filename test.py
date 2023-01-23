from src.path_utils import return_object_from_s3

text = return_object_from_s3("hugo/5_complete_textref.txt")
print(list(set(text)))
