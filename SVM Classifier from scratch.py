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
df['BPMeds'] = df['BPMeds'].fillna(df['BPMeds'].mode()[0])
df['glucose']= df['glucose'].fillna(df['glucose'].median())
df['heartRate']= df['heartRate'].fillna(df['heartRate'].mean())
df['education'] = df['education'].fillna(df['education'].median())
df['cigsPerDay']= df['cigsPerDay'].fillna(df['cigsPerDay'].median())

#drawing boxplot to detect outliers
# plt.figure(figsize=(8,5))
# sns.boxplot(x=df['BMI'])
# plt.title('Boxplot of BMI')
# plt.show()

cols = ['age','cigsPerDay','totChol','sysBP','diaBP','BMI','heartRate','glucose']

for col in cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = df[col].clip(lower, upper)

#After handling outliers
# plt.figure(figsize=(8,5))
# sns.boxplot(x=df['BMI'])
# plt.title('Boxplot of BMI')
# plt.show()

#-------------------------------------------------------------------------
                             #SVC implementation
#-------------------------------------------------------------------------

df= df.values
x, y =df[:, :-1],df[:, -1]
y= np.where(y == 0, -1, 1)#Convert 0 classes to -1
x= (x-x.mean(axis=0)) / (x.std(axis=0)+1e-5)

lr=0.001
lamda=0.01
C=1.0
inters=1000

n_samples, n_features = x.shape
w=np.zeros(n_features)
b=0

for _ in range(inters):

    indices = np.random.permutation(n_samples)#Shuffles data each epoch
    for i in indices:
        xi=x[i]
        yi=y[i]

        margin = yi* (np.dot(xi, w)-b)

        if margin >= 1:
            w -= lr*(2*lamda*w)
        else:
            #Regular. + hinge loss with c
            w -= lr*(2*lamda*w - C*yi*xi)
            b -= lr*(C*yi)

predictions = np.sign(np.dot(x, w) - b)
accuracy = np.mean(predictions == y)

print("Accuracy:", accuracy)