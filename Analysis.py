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

# ============================= Global Sales Code =============================
# - Masking Global Sales 3 STD away from medium
global_sales = np.asarray(data.Global_Sales.tolist())  # millions
global_sales_mean = sp.mean(global_sales)
global_sales_std = sp.std(global_sales)
global_sales = ma.masked_outside(global_sales,
                                 global_sales_mean - global_sales_std * 3,
                                 global_sales_mean + global_sales_std * 3)

# - Regression Critic Score vs Global Sales
slope, intercept, r_value, p_value, std_err = \
    mstats.linregress(critic_score, global_sales)
t_score = slope / std_err

# - Print
with open('results.txt', 'w') as f:
    f.write('Global Sales Regression\nSlope (Million per 1): ' + str(slope) +
            '\nIntercept: ' + str(intercept) + '\nR Value: ' + str(r_value) +
            '\nP Value: ' + str(p_value) + '\nStandard Error: ' + str(std_err)
            + '\nT Stat: ' + str(t_score))
    if t_score > 1.96:
        f.write('; Reject Null Hypothesis\n\n')
    else:
        f.write('; Accept Null Hypothesis\n\n')

# - For plotting regression line
x = np.arange(1, 100)

# - Plotting Global Sales
plt.figure('Global_Sales')
plt.plot(critic_score, global_sales, 'o', markersize=2, label='critic score')
plt.plot(user_score, global_sales, 'or', markersize=2, label='user score')
plt.xlabel('Critic Score/User Score')
plt.ylabel('Global Sales (Millions)')
plt.xlim(0, 100)
plt.ylim(bottom=0)
plt.title('Critic Score/User Score vs. Global Sales')
plt.legend()
# - Plotting regression line
plt.plot(x, intercept + x * slope, 'k')
plt.savefig('Global_Sales')

# =============================== NA Sales Code ===============================
# - Masking NA Sales 3 STD away from medium
NA_sales = np.asarray(data.NA_Sales.tolist())  # millions
NA_sales_mean = sp.mean(NA_sales)
NA_sales_std = sp.std(NA_sales)
NA_sales = ma.masked_outside(NA_sales,
                             NA_sales_mean - NA_sales_std * 3,
                             NA_sales_mean + NA_sales_std * 3)

# - Regression Critic Score vs NA Sales
slope, intercept, r_value, p_value, std_err = \
    mstats.linregress(critic_score, NA_sales)
t_score = slope / std_err

# - Print
with open('results.txt', 'a') as f:
    f.write('NA Sales Regression\nSlope (Million per 1): ' + str(slope) +
            '\nIntercept: ' + str(intercept) + '\nR Value: ' + str(r_value) +
            '\nP Value: ' + str(p_value) + '\nStandard Error: ' + str(std_err)
            + '\nT Stat: ' + str(t_score))
    if t_score > 1.96:
        f.write('; Reject Null Hypothesis\n\n')
    else:
        f.write('; Accept Null Hypothesis\n\n')

# - Plotting NA Sales
plt.figure('NA_Sales')
plt.plot(critic_score, NA_sales, 'o', markersize=2, label='critic score')
plt.plot(user_score, NA_sales, 'or', markersize=2, label='user score')
plt.xlabel('Critic Score/User Score')
plt.ylabel('NA Sales (Millions)')
plt.xlim(0, 100)
plt.ylim(bottom=0)
plt.title('Critic Score/User Score vs. NA Sales')
plt.legend()
# - Plotting regression line
plt.plot(x, intercept + x * slope, 'k')
plt.savefig('NA_Sales')

# =============================== EU Sales Code ===============================
# - Masking EU Sales 3 STD away from medium
EU_sales = np.asarray(data.EU_Sales.tolist())  # millions
EU_sales_mean = sp.mean(EU_sales)
EU_sales_std = sp.std(EU_sales)
EU_sales = ma.masked_outside(EU_sales,
                             EU_sales_mean - EU_sales_std * 3,
                             EU_sales_mean + EU_sales_std * 3)

# - Regression Critic Score vs EU Sales
slope, intercept, r_value, p_value, std_err = \
    mstats.linregress(critic_score, EU_sales)
t_score = slope / std_err

# - Print
with open('results.txt', 'a') as f:
    f.write('EU Sales Regression\nSlope (Million per 1): ' + str(slope) +
            '\nIntercept: ' + str(intercept) + '\nR Value: ' + str(r_value) +
            '\nP Value: ' + str(p_value) + '\nStandard Error: ' + str(std_err)
            + '\nT Stat: ' + str(t_score))
    if t_score > 1.96:
        f.write('; Reject Null Hypothesis\n\n')
    else:
        f.write('; Accept Null Hypothesis\n\n')

# - Plotting EU Sales
plt.figure('EU_Sales')
plt.plot(critic_score, EU_sales, 'o', markersize=2, label='critic score')
plt.plot(user_score, EU_sales, 'or', markersize=2, label='user score')
plt.xlabel('Critic Score/User Score')
plt.ylabel('EU Sales (Millions)')
plt.xlim(0, 100)
plt.ylim(bottom=0)
plt.title('Critic Score/User Score vs. EU Sales')
plt.legend()
# - Plotting regression line
plt.plot(x, intercept + x * slope, 'k')
plt.savefig('EU_Sales')

# =============================== JP Sales Code ===============================
# - Masking JP Sales 3 STD away from medium
JP_sales = np.asarray(data.JP_Sales.tolist())  # millions
JP_sales_mean = sp.mean(JP_sales)
JP_sales_std = sp.std(JP_sales)
JP_sales = ma.masked_outside(JP_sales,
                             JP_sales_mean - JP_sales_std * 3,
                             JP_sales_mean + JP_sales_std * 3)

# - Regression Critic Score vs JP Sales
slope, intercept, r_value, p_value, std_err = \
    mstats.linregress(critic_score, JP_sales)
t_score = slope / std_err

# - Print
with open('results.txt', 'a') as f:
    f.write('JP Sales Regression\nSlope (Million per 1): ' + str(slope) +
            '\nIntercept: ' + str(intercept) + '\nR Value: ' + str(r_value) +
            '\nP Value: ' + str(p_value) + '\nStandard Error: ' + str(std_err)
            + '\nT Stat: ' + str(t_score))
    if t_score > 1.96:
        f.write('; Reject Null Hypothesis\n\n')
    else:
        f.write('; Accept Null Hypothesis\n\n')

# - Plotting JP Sales
plt.figure('JP_Sales')
plt.plot(critic_score, JP_sales, 'o', markersize=2, label='critic score')
plt.plot(user_score, JP_sales, 'or', markersize=2, label='user score')
plt.xlabel('Critic Score/User Score')
plt.ylabel('JP Sales (Millions)')
plt.xlim(0, 100)
plt.ylim(bottom=0)
plt.title('Critic Score/User Score vs. JP Sales')
plt.legend()
# - Plotting regression line
plt.plot(x, intercept + x * slope, 'k')
plt.savefig('JP_Sales')

# ============================= Other Sales Code ==============================
# - Masking other sales 3 STD away from medium
other_sales = np.asarray(data.Other_Sales.tolist())  # millions
other_sales_mean = sp.mean(other_sales)
other_sales_std = sp.std(other_sales)
other_sales = ma.masked_outside(other_sales,
                                other_sales_mean - other_sales_std * 3,
                                other_sales_mean + other_sales_std * 3)

# - Regression Critic Score vs other sales
slope, intercept, r_value, p_value, std_err = \
    mstats.linregress(critic_score, other_sales)
t_score = slope / std_err

# - Print
with open('results.txt', 'a') as f:
    f.write('Other Sales Regression\nSlope (Million per 1): ' + str(slope) +
            '\nIntercept: ' + str(intercept) + '\nR Value: ' + str(r_value) +
            '\nP Value: ' + str(p_value) + '\nStandard Error: ' + str(std_err)
            + '\nT Stat: ' + str(t_score))
    if t_score > 1.96:
        f.write('; Reject Null Hypothesis\n\n')
    else:
        f.write('; Accept Null Hypothesis\n\n')

# - Plotting Other Sales
plt.figure('Other_Sales')
plt.plot(critic_score, other_sales, 'o', markersize=2, label='critic score')
plt.plot(user_score, other_sales, 'or', markersize=2, label='user score')
plt.xlabel('Critic Score/User Score')
plt.ylabel('Other Sales (Millions)')
plt.xlim(0, 100)
plt.ylim(bottom=0)
plt.title('Critic Score/User Score vs. Other Sales')
plt.legend()
# - Plotting regression line
plt.plot(x, intercept + x * slope, 'k')
plt.savefig('Other_Sales')

# ==== end file ====

# TODO: Histograms of data, Mask by year
