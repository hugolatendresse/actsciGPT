from src.path_utils import return_object_from_s3


def get_data(exam="All"):
    exam = str(exam)
    if exam=="All":
        files=[prefix + '_complete_textref.txt' for prefix in ["5","6c","6us","6I","7","8","9"] ]
        text = ""
        for file in files:
            t = return_object_from_s3("hugo/" + file)
            # with open(os.path.join("data", file), 'r', encoding='utf-8') as f:
            #     t = f.read()
            text += t
    else:
        return return_object_from_s3("hugo/" + exam + '_complete_textref.txt')
    return text