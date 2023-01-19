import csv

# open the original train.csv file
with open('../kaggle_data/train.csv', 'r', encoding='utf-8-sig') as original_file:
    # create a new csv file named product_data.csv
    with open('product.csv', 'w', newline='', encoding='utf-8-sig', errors='ignore') as new_file:
        # create a csv reader and writer object
        csv_reader = csv.reader(original_file)
        csv_writer = csv.writer(new_file)
        # read each row and check the first column
        next(csv_reader)  # This line skips the first line of the file
        i = 0
        for row in csv_reader:
            if int(row[0]) == '2':
                row[0] = 2
            elif int(row[0]) == 1:
                row[0] = 1
            csv_writer.writerow([row[0], row[2]])
            i += 1
            if i >= 12000:
                break

# open the original IMDB dataset file
with open('../kaggle_data/IMDB Dataset.csv', 'r', encoding='utf-8-sig') as original_file:
    # create a new csv file named movie_data.csv
    with open('movie.csv', 'w', newline='', encoding='utf-8-sig', errors='ignore') as new_file:
        # create a csv reader and writer object
        csv_reader = csv.reader(original_file)
        csv_writer = csv.writer(new_file)
        # read each row and check the first column
        next(csv_reader)  # This line skips the first line of the file
        i = 0
        for row in csv_reader:
            if row[1] == 'positive':
                row[1] = 2
            elif row[1] == 'negative':
                row[1] = 1
            csv_writer.writerow([row[1], row[0]])
            i += 1
            if i >= 12000:
                break

# open the reviews.csv file
with open('../kaggle_data/reviews.csv', 'r', encoding='utf-8-sig') as original_file:
    # create a new csv file named updated_reviews.csv
    with open('app.csv', 'w', newline='', encoding='utf-8-sig', errors='ignore') as new_file:
        # create a csv reader and writer object
        csv_reader = csv.reader(original_file)
        csv_writer = csv.writer(new_file)
        # read each row and check the first column
        next(csv_reader)  # This line skips the first line of the file
        i = 0
        for row in csv_reader:
            if int(row[4]) == 3:
                continue
            if int(row[4]) > 3:
                row[4] = 2
            elif int(row[4]) < 3:
                row[4] = 1
            csv_writer.writerow([row[4], row[3]])
            i += 1
            if i >= 12000:
                break


# open the reviews.csv file
with open('../kaggle_data/tripadvisor_hotel_reviews.csv', 'r', encoding='utf-8-sig') as original_file:
    # create a new csv file named updated_reviews.csv
    with open('place.csv', 'w', newline='', encoding='utf-8-sig', errors='ignore') as new_file:
        # create a csv reader and writer object
        csv_reader = csv.reader(original_file)
        csv_writer = csv.writer(new_file)
        # read each row and check the first column
        next(csv_reader)  # This line skips the first line of the file
        i = 0
        for row in csv_reader:
            if int(row[1]) == 3:
                continue
            if int(row[1]) > 3:
                row[1] = 2
            elif int(row[1]) < 3:
                row[1] = 1
            csv_writer.writerow([row[1], row[0]])
            i += 1
            if i >= 12000:
                break
