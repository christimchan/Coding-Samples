import pandas
import numpy as np
import re
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import math
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix

''' 
The following is the starting code for path1 for data reading to make your first step easier.
'dataset_1' is the clean data for path1.
'''
dataset_1 = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
dataset_1['Brooklyn Bridge']      = pandas.to_numeric(dataset_1['Brooklyn Bridge'].replace(',','', regex=True))
dataset_1['Manhattan Bridge']     = pandas.to_numeric(dataset_1['Manhattan Bridge'].replace(',','', regex=True))
dataset_1['Queensboro Bridge']    = pandas.to_numeric(dataset_1['Queensboro Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
# print(dataset_1.to_string()) #This line will print out your data

#QUESTION 1
print("Question 1:")

#bridge data
brooklyn = dataset_1['Brooklyn Bridge']
manhattan = dataset_1['Manhattan Bridge']
queensboro = dataset_1['Queensboro Bridge']
williamsburg = dataset_1['Williamsburg Bridge']

#average per bridge
avg_Brook = round(sum(brooklyn) / len(brooklyn))
avg_Man = round(sum(manhattan) / len(manhattan))
avg_Queen = round(sum(queensboro)/ len(queensboro))
avg_Will = round(sum(williamsburg) / len(williamsburg))

#totals
bridge = ['Brooklyn', 'Manhattan', 'Williamsburg', 'Queensboro']
total_avg = [avg_Brook, avg_Man, avg_Will, avg_Queen]

#display the general averages per bridge across all months in dataset (for testing purposes)
count = 0
print("Total Average Bike Traffic Per Bridge in New York City:")

while count < len(bridge):
    print("The average bike traffic on " + bridge[count] + " Bridge is around " + str(round(total_avg[count])) + " cyclists")
    count += 1
print("\n")

#dataset seperated based on month
april = {'Brooklyn Bridge': round(sum(brooklyn[0:30]) / len(brooklyn[0:30])), 'Manhattan Bridge': round(sum(manhattan[0:30]) / len(manhattan[0:30])), 'Williamsburg Bridge': round(sum(williamsburg[0:30]) / len(williamsburg[0:30])), 'Queensboro Bridge': round(sum(queensboro[0:30]) / len(queensboro[0:30]))}
may = {'Brooklyn Bridge': round(sum(brooklyn[30:61]) / len(brooklyn[30:61])), 'Manhattan Bridge': round(sum(manhattan[30:61]) / len(manhattan[30:61])), 'Williamsburg Bridge': round(sum(williamsburg[30:61]) / len(williamsburg[30:61])), 'Queensboro Bridge': round(sum(queensboro[30:61]) / len(queensboro[30:61]))}
june = {'Brooklyn Bridge': round(sum(brooklyn[61:91]) / len(brooklyn[61:91])), 'Manhattan Bridge': round(sum(manhattan[61:91]) / len(manhattan[61:91])), 'Williamsburg Bridge': round(sum(williamsburg[61:91]) / len(williamsburg[61:91])), 'Queensboro Bridge': round(sum(queensboro[61:91]) / len(queensboro[61:91]))}
july = {'Brooklyn Bridge': round(sum(brooklyn[91:122]) / len(brooklyn[91:122])), 'Manhattan Bridge': round(sum(manhattan[91:122]) / len(manhattan[91:122])), 'Williamsburg Bridge': round(sum(williamsburg[91:122]) / len(williamsburg[91:122])), 'Queensboro Bridge': round(sum(queensboro[91:122]) / len(queensboro[91:122]))}
august = {'Brooklyn Bridge': round(sum(brooklyn[122:153]) / len(brooklyn[122:153])), 'Manhattan Bridge': round(sum(manhattan[122:153]) / len(manhattan[122:153])), 'Williamsburg Bridge': round(sum(williamsburg[122:153]) / len(williamsburg[122:153])), 'Queensboro Bridge': round(sum(queensboro[122:153]) / len(queensboro[122:153]))}
september = {'Brooklyn Bridge': round(sum(brooklyn[153:183]) / len(brooklyn[153:183])), 'Manhattan Bridge': round(sum(manhattan[153:183]) / len(manhattan[153:183])), 'Williamsburg Bridge': round(sum(williamsburg[153:183]) / len(williamsburg[153:183])), 'Queensboro Bridge': round(sum(queensboro[153:183]) / len(queensboro[153:183]))}
october = {'Brooklyn Bridge': round(sum(brooklyn[183:214]) / len(brooklyn[183:214])), 'Manhattan Bridge': round(sum(manhattan[183:214]) / len(manhattan[183:214])), 'Williamsburg Bridge': round(sum(williamsburg[183:214]) / len(williamsburg[183:214])), 'Queensboro Bridge': round(sum(queensboro[183:214]) / len(queensboro[183:214]))}

#display averages per month on bridges (for testing purposes)
month = ['April', 'May', 'June', 'July', 'August', 'September', 'October']
month_Values = [list(april.values()), list(may.values()), list(june.values()), list(july.values()), list(august.values()), list(september.values()), list(october.values())]

print("Average Bike Traffic Per Bridge in New York City Per Month")

j = 0
for m in month:
    i = 0

    while i < len(bridge):
        print("The average bike traffic on " + bridge[i] + " Bridge in " + m + " is around " + str(month_Values[j][i]) + " cyclists")
        i += 1
    print("\n")
    j += 1

#making monthly and total graphs (april through october)
plt.figure(1)
plt.bar(bridge, total_avg, color='black')
plt.xticks(np.arange(4), bridge, fontsize = 8)
plt.title('Bike Traffic on Four Bridges in New York City from April through October')
plt.xlabel('Bridge in NY City')
plt.ylabel('Average Daily Bike Traffic')
plt.savefig('total.png')

plt.figure(2)
plt.bar(april.keys(), april.values(), color='red')
plt.xticks(np.arange(4), bridge, fontsize = 8)
plt.title('Bike Traffic on Four Bridges in New York City In April')
plt.xlabel('Bridge in NY City')
plt.ylabel('Average Daily Bike Traffic')
plt.savefig('april.png')

plt.figure(3)
plt.bar(may.keys(), may.values(), color='orange')
plt.xticks(np.arange(4), bridge, fontsize = 8)
plt.title('Bike Traffic on Four Bridges in New York City In May')
plt.xlabel('Bridge in NY City')
plt.ylabel('Average Daily Bike Traffic')
plt.savefig('may.png')

plt.figure(4)
plt.bar(june.keys(), june.values(), color='yellow')
plt.xticks(np.arange(4), bridge, fontsize = 8)
plt.title('Bike Traffic on Four Bridges in New York City In June')
plt.xlabel('Bridge in NY City')
plt.ylabel('Average Daily Bike Traffic')
plt.savefig('june.png')

plt.figure(5)
plt.bar(july.keys(), july.values(), color='green')
plt.xticks(np.arange(4), bridge, fontsize = 8)
plt.title('Bike Traffic on Four Bridges in New York City In July')
plt.xlabel('Bridge in NY City')
plt.ylabel('Average Daily Bike Traffic')
plt.savefig('july.png')

plt.figure(6)
plt.bar(august.keys(), august.values(), color='blue')
plt.xticks(np.arange(4), bridge, fontsize = 8)
plt.title('Bike Traffic on Four Bridges in New York City In August')
plt.xlabel('Bridge in NY City')
plt.ylabel('Average Daily Bike Traffic')
plt.savefig('august.png')

plt.figure(7)
plt.bar(september.keys(), september.values(), color='purple')
plt.xticks(np.arange(4), bridge, fontsize = 8)
plt.title('Bike Traffic on Four Bridges in New York City In September')
plt.xlabel('Bridge in NY City')
plt.ylabel('Average Daily Bike Traffic')
plt.savefig('september.png')

plt.figure(8)
plt.bar(october.keys(), october.values(), color='brown')
plt.xticks(np.arange(4), bridge, fontsize = 8)
plt.title('Bike Traffic on Four Bridges in New York City In October')
plt.xlabel('Bridge in NY City')
plt.ylabel('Average Daily Bike Traffic')
plt.savefig('october.png')

#QUESTION 2
print("Question 2:")

data = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv', thousands = ',')
data.head()

#function to find mean
def mean(num):
    return (sum(num) / len(num))

#seperate dataset into categories
total_Traffic = data['Total'].tolist()
high_Temp = data['High Temp'].tolist()
low_Temp = data['Low Temp'].tolist()
rain = data['Precipitation'].tolist()

#make all precipitation values into floats
for i in range(len(rain)):
    temp = rain[i]
    rain[i] = float(temp)

#function to normalize data
def norm (data):
    result = []
    avg = mean(data)
    std = np.std(np.array(data))

    for i in range(len(data)):
        result.append((data[i] - avg) / std)
    
    return result

#normalize the seperated and catagorized datasets
norm_High_Temp = norm(high_Temp)
norm_Low_Temp = norm(low_Temp)
norm_Rain = norm(rain)

#make column of 1's in feature matrix
ones = []

for i in range(len(total_Traffic)):
    ones.append(1)

#solving the least squares equation to find beta (beta = (X^T * X)^-1 * X^T * y)
combined_data = [norm_High_Temp, norm_Low_Temp, norm_Rain, ones]
X = np.array(combined_data)
y = np.array(total_Traffic)
xT = np.transpose(X)
inv = np.linalg.inv(np.matmul(X, xT))
beta = np.matmul((np.matmul(inv, X)), np.transpose(y))
print(f"beta: {beta}")

#predict new values using model
prediction = np.matmul(xT, np.transpose(beta)).tolist()

#find r^2 value to find coefficient of determination (how good is the fit of the regression to the dataset)
MSE = 0
var = 0
for i in range(len(prediction)):
  MSE += (total_Traffic[i] - prediction[i]) ** 2
  var += (total_Traffic[i] - mean(total_Traffic)) ** 2

r_Squared = 1 - MSE / var
print("r^2 value: ", r_Squared)
print("\n")

#make plots of predicted and actual data to show correlation
plt.figure(9)
plt.scatter(high_Temp, total_Traffic, color='red')
plt.scatter(high_Temp, prediction, color='#fc7703')
plt.title('Total Daily Bike Traffic vs High Temperatures')
plt.xlabel('Temperature (°F)')
plt.ylabel('Total Bike Traffic')
plt.legend(labels=['Actual Traffic', 'Predicted Traffic'])
plt.savefig('High Temps.png')

plt.figure(10)
plt.scatter(low_Temp, total_Traffic, color='purple')
plt.scatter(low_Temp, prediction, color='#6203fc')
plt.title('Total Daily Bike Traffic vs Low Temperatures')
plt.xlabel('Temperature (°F)')
plt.ylabel('Total Bike Traffic')
plt.legend(labels=['Actual Traffic', 'Predicted Traffic'])
plt.savefig('Low Temps.png')

plt.figure(11)
plt.scatter(rain, total_Traffic, color='blue')
plt.scatter(rain, prediction, color='cyan')
plt.title('Total Daily Bike Traffic vs Precipitation')
plt.xlabel('Precipitation (in.)', fontsize=14)
plt.ylabel('Total Bike Traffic')
plt.legend(labels=['Actual Traffic', 'Predicted Traffic'])
plt.savefig('Precipitation.png')

#Question 3
print("Question 3:")

#sort data
brook = data['Brooklyn Bridge']
man = data['Manhattan Bridge']
will = data['Williamsburg Bridge']
queens = data['Queensboro Bridge']
total = data['Total']
days = data['Day']
day = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

#create function that sorts data by day
def daySort (days, data):
    sunday_Total = []
    monday_Total = []
    tuesday_Total = []
    wednesday_Total = []
    thursday_Total = []
    friday_Total = []
    saturday_Total = []

    for i in range(len(days)):
        if(days[i] == 'Sunday'):
            sunday_Total.append(data[i])
        elif(days[i] == 'Monday'):
            monday_Total.append(data[i])
        elif(days[i] == 'Tuesday'):
            tuesday_Total.append(data[i])
        elif(days[i] == 'Wednesday'):
            wednesday_Total.append(data[i])
        elif(days[i] == 'Thursday'):
            thursday_Total.append(data[i])
        elif(days[i] == 'Friday'):
            friday_Total.append(data[i])
        elif(days[i] == 'Saturday'):
            saturday_Total.append(data[i])

    total_Days = [sunday_Total, monday_Total, tuesday_Total, wednesday_Total, thursday_Total, friday_Total, saturday_Total]

    return total_Days

#function that calculates average
def dayAvg (total_Days):
    sunday_avg = round(sum(total_Days[0]) / len(total_Days[0]))
    monday_avg = round(sum(total_Days[1]) / len(total_Days[1]))
    tuesday_avg = round(sum(total_Days[2]) / len(total_Days[2]))
    wednesday_avg = round(sum(total_Days[3]) / len(total_Days[3]))
    thursday_avg = round(sum(total_Days[4]) / len(total_Days[4]))
    friday_avg = round(sum(total_Days[5]) / len(total_Days[5]))
    saturday_avg = round(sum(total_Days[6]) / len(total_Days[6]))
    total_avg = [sunday_avg, monday_avg, tuesday_avg, wednesday_avg, thursday_avg, friday_avg, saturday_avg]

    return total_avg
    
#sort data by total and bridge
total_Day_avg = dayAvg(daySort(days, total))
brooklyn_Day_avg = dayAvg(daySort(days, brook))
manhattan_Day_avg = dayAvg(daySort(days, man))
williamsburg_Day_avg = dayAvg(daySort(days, will))
queensboro_Day_avg = dayAvg(daySort(days, queens))

#print average bike traffic per day (testing)
counter = 0
print("Total Average Bike Traffic Per Day in New York City:")

while counter < len(total_Day_avg):
    print("The average bike traffic on " + day[counter] + "s across all bridges is around " + str(round(total_Day_avg[counter])) + " cyclists")
    counter += 1
print("\n")

#make graphs per bridge and total
plt.figure(12)
plt.bar(day, total_Day_avg, color='black')
plt.xticks(np.arange(7), day, fontsize = 8)
plt.title('Bike Traffic on Four Bridges in New York City from April through October')
plt.xlabel('Day of Week')
plt.ylabel('Average Daily Bike Traffic')
plt.savefig('Total per Day.png')

plt.figure(13)
plt.bar(day, brooklyn_Day_avg, color='red')
plt.xticks(np.arange(7), day, fontsize = 8)
plt.title('Bike Traffic on Brooklyn Bridge from April through October')
plt.xlabel('Day of Week')
plt.ylabel('Average Daily Bike Traffic')
plt.savefig('Brooklyn per Day.png')

plt.figure(14)
plt.bar(day, manhattan_Day_avg, color='orange')
plt.xticks(np.arange(7), day, fontsize = 8)
plt.title('Bike Traffic on Manhattan Bridge from April through October')
plt.xlabel('Day of Week')
plt.ylabel('Average Daily Bike Traffic')
plt.savefig('Manhattan per Day.png')

plt.figure(15)
plt.bar(day, williamsburg_Day_avg, color='green')
plt.xticks(np.arange(7), day, fontsize = 8)
plt.title('Bike Traffic on Williamsburg Bridge from April through October')
plt.xlabel('Day of Week')
plt.ylabel('Average Daily Bike Traffic')
plt.savefig('Williamsburg per Day.png')

plt.figure(16)
plt.bar(day, queensboro_Day_avg, color='blue')
plt.xticks(np.arange(7), day, fontsize = 8)
plt.title('Bike Traffic on Queensboro Bridge from April through October')
plt.xlabel('Day of Week')
plt.ylabel('Average Daily Bike Traffic')
plt.savefig('Queensboro per Day.png')

#normalize total data and add data point to days with less than 31
total_Days = daySort(days, total)
num = 0

while num < len(total_Days):
    if(len(total_Days[num]) == 30):
        total_Days[num].append(total_Day_avg[num])
    num += 1

norm_Sunday = norm(total_Days[0])
norm_Monday = norm(total_Days[1])
norm_Tuesday = norm(total_Days[2])
norm_Wednesday = norm(total_Days[3])
norm_Thursday = norm(total_Days[4])
norm_Friday = norm(total_Days[5])
norm_Saturday = norm(total_Days[6])

norm_Sunday.append(1)
norm_Monday.append(1)
norm_Tuesday.append(1)
norm_Wednesday.append(1)
norm_Thursday.append(1)
norm_Friday.append(1)
norm_Saturday.append(1)

#solve least squares equation
combined_data2 = np.array([norm_Sunday, norm_Monday, norm_Tuesday, norm_Wednesday, norm_Thursday, norm_Friday, norm_Saturday])
X2 = np.transpose(combined_data2)
days_of_Week = [1, 2, 3, 4, 5, 6, 7]
y2 = np.array(days_of_Week)
xT2 = np.transpose(X2)
inv2 = np.linalg.inv(np.dot(X2, xT2))
beta2 = np.matmul((np.matmul(inv2, X2)), np.transpose(y2))
print(f"beta: {beta2}")

#predict new values using model
prediction2 = np.matmul(xT2, np.transpose(beta2)).tolist()
print(prediction2)

#find r^2 value to find coefficient of determination (how good is the fit of the regression to the dataset)
MSE2 = 0
var2 = 0
for i in range(len(prediction2)):
  MSE2 += (days_of_Week[i] - prediction2[i]) ** 2
  var2 += (days_of_Week[i] - mean(days_of_Week)) ** 2

r_Squared2 = 1 - MSE2 / var2
print("r^2 value: ", r_Squared2)
print("\n")

#plot the predictions and actual data to see correlation
plt.figure(17)
plt.scatter(total_Day_avg, days_of_Week, color='red')
plt.scatter(total_Day_avg, prediction2, color='#fc7703')
plt.title('Average Number of Cyclists per Day vs Day of the Week')
plt.xlabel('Average Cyclists')
plt.ylabel('Day of the Week')
plt.legend(labels=['Actual Day', 'Predicted Day'])
plt.savefig('Day Prediction.png')