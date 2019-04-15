import csv

results = []
with open('Video_Game_Sales_as_of_Jan_2017.csv') as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_ALL)
    for row in reader:
        results.append(row)
