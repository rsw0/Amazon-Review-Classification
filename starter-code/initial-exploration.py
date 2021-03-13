import pandas as pd
import matplotlib.pyplot as plt

trainingSet = pd.read_csv("./data/X_train.csv")
testingSet = pd.read_csv("./data/X_submission.csv")

print("train.csv shape is ", trainingSet.shape)
print("test.csv shape is ", testingSet.shape)

print("first 5 rows of the training set")
print(trainingSet.head())
print("column names of the training set")
train_col_names = trainingSet.columns.values.tolist()
print(train_col_names)
print()

print("training set description")
print(trainingSet.describe())

print("first 5 rows of the testing set")
print(testingSet.head())
print("column names of the testing set")
test_col_names = testingSet.columns.values.tolist()
print(test_col_names)
print()


'''
How many of each Score is there in the dataset? How many of each product is there? 
How many distinct users are there? Which users have rated the most products? 
What is the average rating of the top10 most rated products? etc.
What are the top 50 words used in 1 star reviews? 
What is the most popular time of day for 5 star reviews? 
what is the average length of 3 star reviews? etc
'''


print("range of UNIX time spanned")
print("from: " + str(trainingSet["Time"].min()))
print("to: " + str(trainingSet["Time"].max()))

print("changing unix to datetime, then to individual columns")
trainingSet['my_datetime'] = pd.to_datetime(trainingSet['Time'], unit='s')
trainingSet['day'] = trainingSet['my_datetime'].dt.day
trainingSet['month'] = trainingSet['my_datetime'].dt.month
trainingSet['year'] = trainingSet['my_datetime'].dt.year
trainingSet['hour'] = trainingSet['my_datetime'].dt.hour
# once you have it in datetime format, you can extract directly using the dt methods above if you want individual columns
# you don't need the code below
# trainingSet['new_date'] = [d.date() for d in trainingSet['my_datetime']]
# trainingSet['new_time'] = [d.time() for d in trainingSet['my_datetime']]
print("the newly added time rows are")
print(trainingSet.head()[['year','month','day','hour']])

print("number of reviews each year")







# trainingSet['Score'].value_counts().plot(kind='bar', legend=True, alpha=.5)
# plt.show()

# trainingSet['UserId'].value_counts().nsmallest(25).plot(kind='bar', legend=True, alpha=.5)
# plt.show()

# trainingSet[['Score', 'UserId']].groupby('UserId').mean().nsmallest(25, 'Score').plot(kind='bar', legend=True, alpha=.5)
# plt.title('Top 25 kindest Reviewers')
# plt.show()

#trainingSet[trainingSet['ProductID'].isin(trainingSet)['ProductID'].value_counts().nlargest(25).index.tolist]
#plt.title('placeholder')
#plt.show()