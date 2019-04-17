import pandas
import numpy as np

colnames = ['Name', 'Platform', 'Year_Of_Release', 'Genre', 'Publisher',
            'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales',
            'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count',
            'Rating']
data = pandas.read_csv('Video_Game_Sales_as_of_Jan_2017.csv', names=colnames)
name = np.asarray(data.Name.tolist())
platform = np.asarray(data.Platform.tolist())
year_of_release = np.asarray(data.Year_Of_Release.tolist())
genre = np.asarray(data.Genre.tolist())
publisher = np.asarray(data.Publisher.tolist())
NA_sales = np.asarray(data.NA_Sales.tolist())  # millions
EU_sales = np.asarray(data.EU_Sales.tolist())  # millions
JP_sales = np.asarray(data.JP_Sales.tolist())  # millions
other_sales = np.asarray(data.Other_Sales.tolist())  # millions
global_Sales = np.asarray(data.Global_Sales.tolist())  # millions
critic_score = np.asarray(data.Critic_Score.tolist())
critic_count = np.asarray(data.Critic_Count.tolist())
user_score = np.asarray(data.User_Score.tolist())
user_count = np.asarray(data.User_Count.tolist())
rating = np.asarray(data.Rating.tolist())

del colnames
del data
