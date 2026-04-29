import re
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVR
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from narwhals.selectors import Selector
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


#printing first five rows of dataframe
df = pd.read_csv('C:/Users/user/Downloads/Datasets/Datasets train splits/Online Games Popularity Predcition/train_data.csv')
pd.set_option('display.max_columns', None) #As the dataset is large it won't print the whole columns except with these two lines
pd.set_option('display.width', None)
print("DATA'S FIRST FIVE ROWS:\n",df.head())

#dataset info
print("\n\nDATA INFORMATION:")
df.info()

#-------------------------------------------------------------------------
                             #PREPROCESSING
#-------------------------------------------------------------------------

#replace all empty strings with NULL values
df=df.replace(r"^\s*$", np.nan, regex=True)

#checking dataset Duplicates
print("\n\n NUMBER OF DUPLICATED ROWS:",df.duplicated().sum())

#checking dataset NULLS
pd.set_option('display.max_rows', None)
print("\n\n",df.isna().sum())

#Dropping Duplicated Rows
df.drop_duplicates(inplace=True)

#convert ReleaseDate from str to datetime as it's an important feature
df["ReleaseDate"] = pd.to_datetime(df["ReleaseDate"], errors="coerce")
#converting date into numeric value
df["ReleaseDate"] = (df["ReleaseDate"] - df["ReleaseDate"].min()).dt.days
#Handle missing values with median
df["ReleaseDate"] = df["ReleaseDate"].fillna(df["ReleaseDate"].median())

#converting all boolean cols to int cols with TRUE:1, FALSE:0 automatically
bools=df.select_dtypes(include="bool").columns
df[bools] = df[bools].astype(int)

#checking on PriceCurrency values
print("values of ",df['PriceCurrency'].value_counts())
#dropping this column as it has only one value allover the data
df.drop(columns=["PriceCurrency"],inplace=True)

#DRMNotice 11287 99.3% DROPPED
#ExtUserAcctNotice 11222 98.8% DROPPED
#SupportEmail 2990 36.3% DROPPED
#SupportURL 4438  39% DROPPED
#LegalNotice 6665 58.6% DROPPED
#Reviews 8548 75.2% DROPPED
#PCRecReqsText 6410 56.4% DROPPED
#LinuxMinReqsText 8775  77.2% DROPPED
#LinuxRecReqsText 10345 91% DROPPED
#MacMinReqsText 7429 65.4% DROPPED
#MacRecReqsText 9875  86.9% DROPPED
#ShortDescrip 1577 13.8% DROPPED
#Background 611 5.3% DROPPED
#AboutText 583  5.1% DROPPED
#DetailedDescrip 580 5.1% DROPPED
#PCMinReqsText 635 5.59% DROPPED

#Website 2796 24.6% HANDLED
#SupportedLanguages 30 0.26% HANDLED


#some of these column are mendatory to drop them due to high null values that will affect accuracy when filled with median or mean
high_null_cols= ['SupportEmail','SupportURL','ExtUserAcctNotice','DRMNotice','LegalNotice','Reviews','PCRecReqsText',
                 'LinuxMinReqsText','LinuxRecReqsText','MacMinReqsText','MacRecReqsText']
df.drop(columns=high_null_cols,inplace=True)

#converting Website column from str to int websiteLink:1, none:0
df['Website']= df['Website'].notna().astype(int)

#creating new column that has number of languages and dropping SupportedLanguages column
def languages_count(text):
    if pd.isna(text):
        return 0
    text = re.sub(r'\*.*', '', text)  # remove notes
    return len(text.split())
df['num_languages'] = df['SupportedLanguages'].apply(languages_count)
df.drop(columns=['SupportedLanguages'],inplace=True)

#Dropping ShortDescrip column as will use simple models not NLP + it's hard to handle this texts
df.drop(columns=['ShortDescrip'],inplace=True)

#Dropping columns as it's links to images of game which is not effective and can't be handled
df.drop(columns=['Background'],inplace=True)
df.drop(columns=['HeaderImage'],inplace=True)

#other str immutable columns
text_columns = ['AboutText','DetailedDescrip','PCMinReqsText','QueryName','ResponseName']
df.drop(columns=text_columns, inplace=True)

#correlation between target column and all remaining columns
corr_matrix=df.corr()
target_corr=corr_matrix['RecommendationCount'].sort_values(ascending=False)
fig = px.bar(target_corr,title="Feature Correlation with RecommendationCount")
fig.show()

# fig = px.scatter_matrix(
#     df,
#     dimensions=['SteamSpyOwnersVariance', 'SteamSpyPlayersVariance', 'SteamSpyOwners', 'SteamSpyPlayersEstimate',  'RecommendationCount']
# )
# fig.show()

#SCATTER PLOT FOR ALL EFFECTIVE FEATURES
# features = ['SteamSpyOwnersVariance', 'SteamSpyPlayersVariance', 'SteamSpyOwners', 'SteamSpyPlayersEstimate', ]  # change this
# for col in features:
#     fig = px.scatter(df, x=col, y='RecommendationCount', title=f'{col} vs RecommendationCount')
#     fig.show()


X = df.drop(columns=['RecommendationCount'])
y = df['RecommendationCount']
#ANOVA
selector_f = SelectKBest(score_func=f_regression, k=10)
selector_f.fit(X, y)
f_features = X.columns[selector_f.get_support()]
#Mutual INfo
selector_mi = SelectKBest(score_func=mutual_info_regression, k=10)
selector_mi.fit(X, y)
mi_features = X.columns[selector_mi.get_support()]
#Lasso
model=Lasso(alpha=0.01)
model.fit(X, y)
lasso_features = X.columns[model.coef_!= 0]

print("ANOVA:", list(f_features))
print("Mutual Info:", list(mi_features))
print("Lasso:", list(lasso_features))

'''
from the results of "correlation matrix, Lasso regressor, Mutual Information, ANOVA F-test"
we took the features that were common in these four features selectors and correlation matrix
theses features are "QuerryID, ResponseID, Metacritic, SteamSpyOwners, SteamSpyOwnersVariance, SteamSpyPlayersVariance,
SteamSpyPlayersEstimate,AchievementCount, CategoryInAppPurchase2, ReleaseDate2,  PriceInitial2, moviecount2"
'''
newdf_list=['RecommendationCount','QueryID','ResponseID','Metacritic','SteamSpyOwners','SteamSpyOwnersVariance',
        'SteamSpyPlayersVariance','SteamSpyPlayersEstimate','AchievementCount',
        'CategoryInAppPurchase','ReleaseDate','PriceInitial','MovieCount']

new_df = df[newdf_list]

# Scatter plots for top 10 features with log scale
top_features = ['SteamSpyOwners', 'SteamSpyOwnersVariance',
                'SteamSpyPlayersVariance', 'SteamSpyPlayersEstimate',
                'AchievementCount', 'Metacritic', 'MovieCount',
                'ReleaseDate', 'PriceInitial', 'CategoryInAppPurchase']

for feature in top_features:
    fig = px.scatter(
        new_df,
        x=feature,
        y='RecommendationCount',
        title=f'{feature} vs RecommendationCount',
        opacity=0.5,
        log_x=True,
        log_y=True,
        trendline='ols'
    )
    fig.show()
#-------------------------------------------------------------------------
#                             #MODELLING
#-------------------------------------------------------------------------

#splitting data
data=df[top_features + ['RecommendationCount']].copy()
data['target']=np.log1p(data['RecommendationCount'])

x= data[top_features]
y= data['target']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#scaling the data for better results
scaler=StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# cross validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)


def check_model(name, model):
    r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
    rmse = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
    mae= cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')
    print("\n", name)
    print("R2:", r2.mean())
    print("RMSE:", -rmse.mean())
    print("MAE:", -mae.mean())

print("\n===== CROSS VALIDATION =====")

#simple linear regression
check_model("Linear Regression", LinearRegression())

#Polynomial regression + ridge to prevent overfitting
check_model("Polynomial Regression (degree 2)",
    Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('ridge', Ridge(alpha=10000)) ]))

#SVR
check_model("SVR (RBF kernel)", SVR(kernel='rbf', C=50, epsilon=0.5))

#tuning polynomial ridge strength
best_alpha= None
best_score= -999

for alpha in [100, 1000, 5000, 10000, 20000]:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('ridge', Ridge(alpha=alpha))
    ])

    score = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2').mean()
    print("alpha:", alpha, "-> R2:", score)

    if score > best_score:
        best_score = score
        best_alpha = alpha

print("best alpha found:", best_alpha)

#tuning SVR
best_params = None
best_score = -999

for C in [10, 50]:
    for eps in [0.05, 0.1, 0.5, 1.0]:

        model = SVR(kernel='rbf', C=C, epsilon=eps)
        score = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2').mean()

        print("C:", C, "epsilon:", eps, "-> R2:", score)

        if score > best_score:
            best_score = score
            best_params = (C, eps)

print("best SVR params:", best_params)
# -------------------------------------------------------------------------
                         #Regression Lines Plotting
# -------------------------------------------------------------------------
def plot_regression_results(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    plt.figure(figsize=(10, 6))

    sns.scatterplot(x=y_test, y=y_pred, alpha=0.4)

    line_coords = np.linspace(y_test.min(), y_test.max(), 100)
    plt.plot(line_coords, line_coords, color='red', linestyle='--', label='Perfect Prediction')

    plt.title(f'Regression Results: {name}')
    plt.xlabel('Actual Recommendations')
    plt.ylabel('Predicted Recommendations')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()


plot_regression_results("Linear Regression", LinearRegression().fit(X_train, y_train), X_test, y_test)

plot_regression_results("Polynomial Ridge", Pipeline(
                            [('poly', PolynomialFeatures(degree=2, include_bias=False)),
                             ('ridge', Ridge(alpha=best_alpha))]).fit(X_train, y_train), X_test, y_test)

plot_regression_results("SVR (RBF)",
                        SVR(kernel='rbf', C=best_params[0], epsilon=best_params[1]).fit(X_train, y_train), X_test, y_test)