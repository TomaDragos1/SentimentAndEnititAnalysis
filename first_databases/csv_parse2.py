import csv

# open the train_dataset.csv file
with open('train_dataset.csv', 'w', newline='', encoding='utf-8', errors='replace') as train_file:
    # create a csv writer object
    csv_writer = csv.writer(train_file)
    # write the header row
    csv_writer.writerow(['Sentiment', 'Text', 'Entity'])

    # open the test_dataset.csv file
    with open('test_dataset.csv', 'w', newline='', encoding='utf-8', errors='replace') as test_file:
        # create a csv writer object
        test_writer = csv.writer(test_file)
        # write the header row
        test_writer.writerow(['Sentiment', 'Text', 'Entity'])

        # open each input file
        for file_name in ['app.csv', 'movie.csv', 'place.csv', 'product.csv']:
            with open(file_name, 'r', errors='replace') as input_file:
                # create a csv reader object
                csv_reader = csv.reader(input_file)
                # skip the header row
                next(csv_reader)
                i = 0
                # iterate over the rows
                for row in csv_reader:
                    if i < 8000:
                        csv_writer.writerow([row[0], row[1], file_name.split(".")[0]])
                    else:
                        test_writer.writerow([row[0], row[1], file_name.split(".")[0]])
                    i += 1
