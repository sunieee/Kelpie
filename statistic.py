import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, brunnermunzel
import argparse
from sklearn.metrics import r2_score

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, default='output/FB15k-237/ConvE')

score_file = parser.parse_args().folder + '/score_df.csv'
df = pd.read_csv(score_file)
df_des = df.describe()
df_des.to_csv(parser.parse_args().folder + '/score_df_describe.csv')
print(df_des)

'''
I have two dataframe: domain[feature], other[feature], representing two types of dataset, they all have the same columns: 'feature'. I want to:
1. examine whether each column in two dataframe matches Normal distribution and has same variance. If so, apply t-test
2. examine whether each column in two dataframe has same variance. If so, apply Mann-Whitney U test
3. For columns do not match the above requirements, apply Brunner-Munzel test
return the column whose test is significant (pvalue<0.05 for t-test) and the test method respectively
'''

def test(series1, series2):
    # check if the feature follows normal distribution in both dataframes
    _, p1 = shapiro(series1)
    _, p2 = shapiro(series2)
    
    # check if the variance is equal in both dataframes
    _, p3 = levene(series1, series2)
    
    # apply t-test if the feature follows normal distribution and has equal variance
    if p1 > 0.05 and p2 > 0.05 and p3 > 0.05:
        _, p = ttest_ind(series1, series2)
        test_method = 't-test'
    # apply Mann-Whitney U test if the feature does not follow normal distribution but has equal variance
    elif p1 <= 0.05 and p2 <= 0.05 and p3 > 0.05:
        _, p = mannwhitneyu(series1, series2)
        test_method = 'Mann-Whitney U test'
    # apply Brunner-Munzel test if the variance is not equal
    else:
        _, p = brunnermunzel(series1, series2)
        test_method = 'Brunner-Munzel test'
    
    return p, test_method


# create an empty dictionary to store the p-value and test method for each feature
print(test(df['truth'], df['approx']))


print('R^2', r2_score(df['truth'], df['approx']))

df_length1 = df[df['length'] == 1]
print('R^2(length=1)', r2_score(df_length1['truth'], df_length1['approx']))

df_length2 = df[df['length'] == 2]
print('R^2(length=2)', r2_score(df_length2['truth'], df_length2['approx']))

'''
To calculate the R^2 value, or the coefficient of determination, 
you need a model to predict your dependent variable from your independent variable(s).
 R^2 is a measure of how well the model's predictions match the actual values.
R^2 = 1 - (sum of squared residuals / total sum of squares)

(0.0, 'Brunner-Munzel test')
R^2 -0.2616862976049872
R^2(length=1) -0.027197611036207325
R^2(length=2) -0.7255528247138718
'''