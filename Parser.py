# =============================================================================
#                           General Documentation
"""
Plots and runs regression on critic scores vs. video game sales
"""

# =============================================================================
#                         Additional Documentation
# Modification History:
# - Apr 2019: Original by Tyler Larican, CSSE Undergraduate,
#   University of Washington Bothell.
# Notes:
# - Written for Python 3.7
# -----------------------------------------------------------------------------


# ------------------ Module General Import and Declarations -------------------
import pandas
import numpy as np
import scipy as sp
import scipy.stats.mstats as mstats
import numpy.ma as ma
import matplotlib.pyplot as plt

# ---------------------------------- Program ----------------------------------
colnames = ['Name', 'Platform', 'Year_Of_Release', 'Genre', 'Publisher',
            'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales',
            'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count',
            'Rating']
# - Use Pandas to read columns according to colnames
data = pandas.read_csv('Video_Game_Sales_as_of_Jan_2017.csv', names=colnames)

# - Converting columns into NumPy Arrays
name = np.asarray(data.Name.tolist())
platform = np.asarray(data.Platform.tolist())
year_of_release = np.asarray(data.Year_Of_Release.tolist())
genre = np.asarray(data.Genre.tolist())
publisher = np.asarray(data.Publisher.tolist())

# - Masking NA Sales 3 STD away from medium
NA_sales = np.asarray(data.NA_Sales.tolist())  # millions
NA_sales_mean = sp.mean(NA_sales)
NA_sales_std = sp.std(NA_sales)
NA_sales = ma.masked_outside(NA_sales,
                             NA_sales_mean - NA_sales_std * 3,
                             NA_sales_mean + NA_sales_std * 3)

EU_sales = np.asarray(data.EU_Sales.tolist())  # millions
JP_sales = np.asarray(data.JP_Sales.tolist())  # millions
other_sales = np.asarray(data.Other_Sales.tolist())  # millions

# - Masking Global Sales 3 STD away from medium
global_sales = np.asarray(data.Global_Sales.tolist())  # millions
global_sales_mean = sp.mean(global_sales)
global_sales_std = sp.std(global_sales)
global_sales = ma.masked_outside(global_sales,
                                 global_sales_mean - global_sales_std * 3,
                                 global_sales_mean + global_sales_std * 3)

# - Masking Critic Scores with review count less than 20
critic_score = np.asarray(data.Critic_Score.tolist())
critic_count = ma.masked_invalid(data.Critic_Count.tolist())
critic_count = ma.masked_less(critic_count, 20)
critic_score = ma.masked_object(critic_score, critic_count)

# - Masking User Scores with review count less than 20
user_score = np.asarray(data.User_Score.tolist()) * 10
user_count = ma.masked_invalid(data.User_Count.tolist())
user_count = ma.masked_less(user_count, 20)
user_score = ma.masked_object(user_score, user_count)

rating = np.asarray(data.Rating.tolist())

# - Unneeded data unallocated
del colnames
del data

# - Plotting Global Sales
plt.figure()
plt.plot(critic_score, global_sales, 'o', markersize=2, label='critic score')
plt.plot(user_score, global_sales, 'or', markersize=2, label='user score')
plt.xlabel('Critic Score/User Score')
plt.ylabel('Global Sales (Millions)')
plt.title('Critic Score/User Score vs. Global Sales')
plt.show()

# - Regression Critic Score vs Global Sales
slope, intercept, r_value, p_value, std_err = mstats.linregress(critic_score, global_sales)

# - Printing regression
print('Global Sales Regression')
print('Slope: ' + str(slope) + '\nIntercept: ' + str(intercept) + '\nR Value: '
      + str(r_value) + '\nP Value: ' + str(p_value) + '\nStandard Error: '
      + str(std_err))

# - Plotting NA Sales
plt.figure()
plt.plot(critic_score, NA_sales, 'o', markersize=2, label='critic score')
plt.plot(user_score, NA_sales, 'or', markersize=2, label='user score')
plt.xlabel('Critic Score/User Score')
plt.ylabel('NA Sales (Millions)')
plt.title('Critic Score/User Score vs. NA Sales')
plt.show()

# ==== end file ====
