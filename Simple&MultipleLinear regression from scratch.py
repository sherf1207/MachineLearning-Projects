import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

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

#feature engineering(making new column that is the multiplication of appliances used and no. of occupants and place it before the last column)
df.insert(len(df.columns)- 1, "Occupant_Appliance_Load", df["Appliances Used"]* df["Number of Occupants"])

#checking for outliers
'''
plot_df=pd.melt(df)
fig=px.box(plot_df, x='variable',y='value', title='Box Plot')
fig.show()+ 
'''
#outliers appeared only in the new column

#removing outliers in new column
Q1=df["Occupant_Appliance_Load"].quantile(0.25)
Q3=df["Occupant_Appliance_Load"].quantile(0.75)
IQR=Q3-Q1
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR
df["Occupant_Appliance_Load"]=df["Occupant_Appliance_Load"].clip(lower=lower_bound, upper=upper_bound)

#plotting the correlation matrix and the heatmap
'''
plt.figure(figsize=(14, 10))  # adjust the figure size
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap of Dataset Features')
plt.show()
'''
#as we can see new column has high correlation with target column, now as dataset is cleas we will dive right into building the models

#----------------------------------------------------------
#SIMPLE LINEAR REGRESSION BUILDING
#----------------------------------------------------------

y=df["Energy Consumption"].to_numpy()
columns=["Square Footage","Number of Occupants","Appliances Used","Average Temperature"]
x1=df[columns].to_numpy()
x2=df[["Number of Occupants","Appliances Used"]].to_numpy()
lr=[0.001, 0.0025, 0.005, 0.01]
epochs=[100,250,500,1000]


def simpleLinearRegression(x,y,lr,epochs):
    x=(x-np.mean(x))/(np.std(x))
    m=0
    c=0
    n=len(x)
    for i in range(epochs):
        ypred=m*x+c
        dm= (-2*np.sum(x*(y-ypred)))/n
        dc= (-2*np.sum(y-ypred))/n

        m=m-lr*dm
        c=c-lr*dc
    ypred = m * x + c
    mse=np.mean((y-ypred)**2)
    return mse

#----------------------------------------------------------
#MULTI LINEAR REGRESSION BUILDING
#----------------------------------------------------------
def multiLinearRegression(x,y,lr,epochs):
    x=(x-np.mean(x, axis=0))/(np.std(x, axis=0) )
    n,features = x.shape
    m=np.zeros(features)
    c=0
    for i in range(epochs):
        ypred = np.dot(x,m)+c
        dm= (-2*(np.dot(x.T,y-ypred)))/n
        dc= (-2*np.sum(y-ypred))/n

        m = m - lr * dm
        c = c - lr * dc
    ypred = np.dot(x, m) + c
    mse = np.mean((y - ypred) ** 2)
    return mse

#----------------------------------------------------------
#Calculating MSE While Changing LR
#----------------------------------------------------------
mse_lr_list={model:[] for model in columns+["All Features", "Occupants+Appliances"]}
for i in columns:
    x=df[i].to_numpy()
    for j in lr:
        mse= simpleLinearRegression(x,y,j,500)
        mse_lr_list[i].append(mse)
        print(f"Feature {i}, MSE: {mse}")

for i in lr:
    mse1=multiLinearRegression(x1, y, i, 500)
    mse2=multiLinearRegression(x2, y, i, 500)
    mse_lr_list["All Features"].append(mse1)
    mse_lr_list["Occupants+Appliances"].append(mse2)
    print(f"All Features MSE: {mse1}")
    print(f"Number of Occupants+Appliances Used MSE: {mse2}")

#----------------------------------------------------------
#LR PLOTTING
#----------------------------------------------------------
plt.figure(figsize=(8,6))
for model, mse_vals in mse_lr_list.items():
    plt.plot(lr, mse_vals, marker='o', label=model)
plt.xlabel("Learning Rate")
plt.ylabel("MSE Loss")
plt.title("Learning Rate vs MSE")
plt.legend()
plt.show()


#----------------------------------------------------------
#Calculating MSE While Changing LR
#----------------------------------------------------------
mse_epoch_list = {model:[] for model in columns+["All Features", "Occupants+Appliances"]}

for i in columns:
    x=df[i].to_numpy()
    for j in epochs:
        mse= simpleLinearRegression(x,y,0.003,j)
        mse_epoch_list[i].append(mse)
        print(f"Feature {i}, MSE: {mse}")

for i in epochs:
    mse1=multiLinearRegression(x1, y, 0.003, i)
    mse2=multiLinearRegression(x2, y, 0.003, i)
    mse_epoch_list["All Features"].append(mse1)
    mse_epoch_list["Occupants+Appliances"].append(mse2)
    print(f"All Features MSE: {mse1}")
    print(f"Number of Occupants+Appliances Used MSE: {mse2}")
#----------------------------------------------------------
#EPOCHS PLOTTING
#----------------------------------------------------------
plt.figure(figsize=(8,6))
for model, mse_vals in mse_epoch_list.items():
    plt.plot(epochs, mse_vals, marker='o', label=model)
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Epochs vs MSE")
plt.legend()
plt.show()
