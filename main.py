
from src.get_actsci_data import get_data


def main():
    text = get_data(exam="9")
    print(str(round(len(text)/10**6,1))+"M")


if __name__ == '__main__':
    main()
