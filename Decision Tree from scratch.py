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
                    #Decision Tree classifier implementation
#-------------------------------------------------------------------------
df= df.values

x=df[:, :-1]
y=df[:, -1]

def tree_model(x, y, depth=0, max_depth=3):
    if len(set(y)) == 1:
        return y[0]

    if depth >= max_depth:
        return max(set(y), key=list(y).count)

    best_f, best_t, best_g = None, None, float('inf')

    n, m = x.shape

    for f in range(m):
        for t in set(x[:, f]):
            left= x[:, f] <= t
            right= x[:, f] > t

            if sum(left) == 0 or sum(right) == 0:
                continue

            yl, yr = y[left], y[right]

            gl= 1 - sum((np.sum(yl == c)/len(yl))**2 for c in set(yl))
            gr= 1 - sum((np.sum(yr == c)/len(yr))**2 for c in set(yr))

            g= (len(yl)/len(y))*gl + (len(yr)/len(y))*gr

            if best_g > g:
                best_f, best_t, best_g = f, t, g

    if best_f is None:   #Return majority class if no split improved
        return max(set(y), key=list(y).count)

    left= x[:, best_f] <= best_t
    right= x[:, best_f] > best_t

    return {
        "f": best_f,
        "t": best_t,
        "l": tree_model(x[left], y[left], depth+1, max_depth),
        "r": tree_model(x[right], y[right], depth+1, max_depth)
    }


tree= tree_model(x, y, max_depth=3)


correct = 0
for i in range(len(x)):
    node=tree
    while isinstance(node, dict):
        if x[i][node["f"]] <= node["t"]:
            node=node["l"]
        else:
            node=node["r"]

    if node == y[i]:
        correct+=1

print("Accuracy:", (correct/len(x))*100,"%")