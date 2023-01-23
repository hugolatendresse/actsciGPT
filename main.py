import os


def main():
    import numpy
    import pandas
    files=[prefix + '_complete_textref.txt' for prefix in ["5","6c","6us","6I","7","8","9"] ]
    text = ""
    for file in files:
        with open(os.path.join("data", file), 'r', encoding='utf-8') as f:
            t = f.read()
        text += t
    print(len(text))




if __name__ == '__main__':
    main()