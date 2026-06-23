# data --> data preprocessing --> data analysis --> Train test split --> Logistic regression modal -->Evalution
# importing Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# data collection and processing
# load the data from csv file to Pandas DataFrame
titanic_data = pd.read_csv(r'C:\Users\allen\OneDrive\Desktop\understanding\data\titanic data\train.csv')

# printing the first 5 rows of the dataframe
print(titanic_data.head())

# number of rows and Columns
print(titanic_data.shape)

# getting some information about data
print(titanic_data.info())

print(titanic_data.isnull().sum())


# drop the "Cabin" column from the dataframe
titanic_data = titanic_data.drop(columns='Cabin')

# replacing the missing values in "Age" column with mean value
titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].mean())

# finding the mode value of "Embarked" column
print(titanic_data['Embarked'].mode())

# replacing the null values in 'Embarked' column with mode values

titanic_data['Embarked'] = titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0])
titanic_data['Fare'] = titanic_data['Fare'].fillna(titanic_data['Fare'].median())

print(titanic_data.isnull().sum())

# data analysis
# getting some statical information
print(titanic_data.describe())


# find the number of people survived or not in titanic

titanic_data['Survived'].value_counts()

sns.countplot(x='Survived', data=titanic_data)
plt.show()

titanic_data['Sex'].value_counts()
plt.show()

# making a count plot for "Sex" column
sns.countplot(x='Sex', data=titanic_data)
plt.show()
# number of survivors Gender wise
sns.countplot(x='Sex', hue='Survived', data=titanic_data)
plt.show()
# making a count plot for "Pclass" column
sns.countplot(x='Pclass', data=titanic_data)
plt.show()
sns.countplot(x='Pclass', hue='Survived', data=titanic_data)
plt.show()


# encoding the categorical columns
# converting categorical columns
# Change line 69 to this:
titanic_data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)
print(titanic_data.head())

# seperate features and target
X = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'])
Y = titanic_data['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

model = LogisticRegression()

# training the Logistic Regression model with training data
model.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)

training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)

test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)
