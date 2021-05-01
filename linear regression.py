import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('victims.csv')
# print(df.shape)
years = pd.DataFrame(df['year'])
cases = pd.DataFrame(df['cases'])
lm = linear_model.LinearRegression()
model = lm.fit(years, cases)
print(df)
predictedItem = model.predict([[2014]])
print("Rape cases in 2014:",predictedItem)
predictedItem = model.predict([[2015]])
print("Rape cases in 2015:",predictedItem)
predictedItem = model.predict([[2016]])
print("Rape cases in 2016:",predictedItem)
predictedItem = model.predict([[2017]])
print("Rape cases in 2017:",predictedItem)
predictedItem = model.predict([[2018]])
print("Rape cases in 2018:",predictedItem)
predictedItem = model.predict([[2019]])
print("Rape cases in 2019:",predictedItem)
predictedItem = model.predict([[2020]])
print("Rape cases in 2020:",predictedItem)
predictedItem = model.predict([[2021]])
print("Rape cases in 2021:",predictedItem)
predictedItem = model.predict([[2022]])
print("Rape cases in 2022:",predictedItem)
predictedItem = model.predict([[2023]])
print("Rape cases in 2023:",predictedItem)
predictedItem = model.predict([[2024]])
print("Rape cases in 2024:",predictedItem)
predictedItem = model.predict([[2025]])
print("Rape cases in 2025:",predictedItem)
"""
=== OUTPUT ===
   year  cases
0  2014   2669
1  2015   2509
2  2016   2938
3  2017   3445
4  2018   3832
5  2019   3881
Rape cases in 2014: [[2459.76190476]]
Rape cases in 2015: [[2760.79047619]]
Rape cases in 2016: [[3061.81904762]]
Rape cases in 2017: [[3362.84761905]]
Rape cases in 2018: [[3663.87619048]]
Rape cases in 2019: [[3964.9047619]]
Rape cases in 2020: [[4265.93333333]]
Rape cases in 2021: [[4566.96190476]]
Rape cases in 2022: [[4867.99047619]]
Rape cases in 2023: [[5169.01904762]]
Rape cases in 2024: [[5470.04761905]]
Rape cases in 2025: [[5771.07619048]]
"""