import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#import plotly.express as px


#printing first five rows of dataframe
df = pd.read_csv('C:/Users/user/Downloads/Assignment1/Assignment1/assignment1dataset.csv')
print("DATA'S FIRST FIVE ROWS:\n",df.head())

#dataset info
print("\n\nDATA INFORMATION:")
df.info()

#----------------------------------------------------------
#PREPROCESSING
#----------------------------------------------------------

#now we have to check for dataset NULLS and Duplicates
print("\n\n NUMBER OF DUPLICATED ROWS:",df.duplicated().sum())
print("\n\n",df.isna().sum())
#no duplictes or NULLS in this dataset

#checking for outliers
'''
plot_df=pd.melt(df)
fig=px.box(plot_df, x='variable',y='value', title='Box Plot')
fig.show()
'''
#No outliers detected

#plotting the correlation matrix and the heatmap
corr=df.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

'''
we're gonna use "Square footage","number of occupants" as they have 
the highest correlation with target column with "square footage" on top of them,
now as dataset is cleas we will dive right into building the model
'''
x= df[['Square Footage','Number of Occupants']]
y= df['Energy Consumption']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

def createFeatures(x, degree):
    x_poly= x.copy()

    a= x.iloc[:,0]
    b= x.iloc[:,1]

    for i in range(2, degree+1):
        x_poly[f'a^{i}']= a**i
        x_poly[f'b^{i}']= b**i
        for j in range(1,degree):
            x_poly[f'a^{j}b^{degree-j}']= (a**j)*(b**(degree-j))
    return x_poly

trainError= []
testError= []

#trying random degree as it's not specified
degrees= range(1, 9)
for degree in degrees:
    #transform features
    xTrain_poly=createFeatures(x_train,degree)
    xTest_poly=createFeatures(x_test,degree)

    #taraining the model
    model= LinearRegression()
    model.fit(xTrain_poly,y_train)

    #prediction
    trainPred= model.predict(xTrain_poly)
    testPred= model.predict(xTest_poly)

    #calculating MSE and append it in list
    trainMSE= mean_squared_error(y_train,trainPred)
    testMSE= mean_squared_error(y_test,testPred)
    trainError.append(trainMSE)
    testError.append(testMSE)

    print(f"Degree {degree}: Train MSE = {trainMSE:.2f}, Test MSE = {testMSE:.2f}")

plt.plot(degrees,trainError, marker='o', label= "Train Error")
plt.plot(degrees,testError, marker='o', label= "Test Error")
plt.xlabel("Degree")
plt.ylabel("MSE")
plt.title("Degree vs Error")
plt.grid(True)
plt.legend()
plt.show()