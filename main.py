import csv

def main():
    # dataFile = open("data.csv", "r")
    # dataContent = csv.DictReader(dataFile)

    # dataFile.close()
    # # x = dataContent
    # # y = dataContent[1]
    # print(dataContent)
    file_path = 'data.csv'
    try:
        with open(file_path, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                print(f'\tCar have {row["km"]} Km and the price is {row["price"]}.')
                line_count += 1
            print(f'Processed {line_count} lines.')
    except FileNotFoundError:
        print(f'The file {file_path} does not exist.')
    except Exception as e:
        print(f"An error occurred: {e} is not in csv")
    print(csv_reader)

main()