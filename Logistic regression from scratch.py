import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('C:/Users/user/OneDrive/Attachments/Desktop/datasets/framingham.csv')
print(df.head())

#-------------------------------------------------------------------------
                             #PREPROCESSING
#-------------------------------------------------------------------------

df.info()
#The whole dataset is already int&float we won't convert anything
print("Number of duplicated rows= ",df.duplicated().sum())
#There're zero duplicated row
print(df.isnull().sum())
#Filling the nulls with mean, median & mode

df['BMI']= df['BMI'].fillna(df['BMI'].mean())
df['totChol']=df['totChol'].fillna(df['totChol'].mean())
df['glucose']= df['glucose'].fillna(df['glucose'].median())
df['heartRate']= df['heartRate'].fillna(df['heartRate'].mean())
df['cigsPerDay']= df['cigsPerDay'].fillna(df['cigsPerDay'].median())

#drawing boxplot to detect outliers
plt.figure(figsize=(8,5))
sns.boxplot(x=df['BMI'])
plt.title('Boxplot of BMI')
plt.show()

cols = ['age','cigsPerDay','totChol','sysBP','diaBP','BMI','heartRate','glucose']

for col in cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = df[col].clip(lower, upper)

#After handling outliers
plt.figure(figsize=(8,5))
sns.boxplot(x=df['BMI'])
plt.title('Boxplot of BMI')
plt.show()

#-------------------------------------------------------------------------
                    #Logistic Regression implementation
#-------------------------------------------------------------------------

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#Splitting the data train 80%/ test 20%
split = int(0.8 * len(x))
Xtrain, Xtest = x[:split], x[split:]
Ytrain, Ytest = y[:split], y[split:]

samples, features= Xtrain.shape
weights= np.zeros(features)

bias=0
lr=0.1
num_of_iterations=1000

#Training loop
for _ in range(num_of_iterations):
    z= np.dot(Xtrain, weights)+ bias

    #Sigmoid function
    ypred= 1 / (1 + np.exp(-z))

    #Gradient
    dw=(1 / samples) * np.dot(Xtrain.T, (ypred- Ytrain))
    db=(1 / samples) * np.sum(ypred- Ytrain)

    weights-= lr*dw
    bias-= lr*db

z_test = np.dot(Xtest, weights) + bias
ypred_test= 1 / (1+np.exp(-z_test))

#Converts probs to classes
labels= np.where(ypred_test >=0.5,1,0)
accuracy= np.mean(labels==Ytest)
print("Accuracy:", accuracy)

