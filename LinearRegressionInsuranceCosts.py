#%%
# Import libraries
import mlflow
import pandas  as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#%%
# Read in data
df = pd.read_csv("insurance.csv")

# Look into top few rows and columns in the dataset
df.head()

#%% Plotting BMI
sns.lmplot(x='bmi',y='charges',data=df,aspect=2,height=6)
plt.xlabel('BMI:')
plt.ylabel('Insurance Charges:')
plt.title('Charge Vs BMI')


# %%
# correlation plot
corr = df.corr()
sns.heatmap(corr, cmap = 'Wistia', annot= True)


#%%
# Reduce dataset
df = df.drop(columns=["sex", "region"])

# %%
# Encoding
le = preprocessing.LabelEncoder()
le.fit(df["smoker"])
encoded_lables = le.transform(df["smoker"])

df["smoker"] = encoded_lables

# Inverse
#list(le.inverse_transform(encoded_lables))

# %%
# Create train and test split
X = df.drop('charges',axis=1) # Independet variable
y = df['charges'] # dependent variable

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=23)


# %%
# Train the sklearn model and implement logging with mlflow
with mlflow.start_run():

    mlflow.set_tag("mlflow.runName", "Tech-Talk-02") 
    
    lin_reg = LinearRegression()
    lin_reg.fit(X_train,y_train) 

    mlflow.sklearn.log_model(lin_reg, "lr_model")

    """
    R-Squared (R² or the coefficient of determination) is a statistical measure in a
    regression model that determines the proportion of variance in the dependent variable
    that can be explained by the independent variable. In other words, r-squared shows 
    how well the data fit the regression model (the goodness of fit)
    """
    mlflow.log_param("R_sq", lin_reg.score(X_test, y_test))

    mlflow.log_param("Intercept", lin_reg.intercept_)
    mlflow.log_param("Theta_age", lin_reg.coef_[0])
    mlflow.log_param("Theta_smoker", lin_reg.coef_[3])

    corr = df.corr()
    sns.heatmap(corr, cmap = 'Wistia', annot= True)

    heatmap_figure = sns.heatmap(corr, cmap = 'Wistia', annot= True).get_figure()
    heatmap_figure.savefig('heatmap.png', dpi=400)

    mlflow.log_artifact('heatmap.png')


# %%
# Inference with logged mlflow model
logged_model = 'runs:/1db694d0b5df45bfae95abc79a884e4d/lr_model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

age = 18
bmi = 33
children = 1
smoker = 1

charge = round(loaded_model.predict([[age, bmi, children, smoker]])[0],2)
print(str(charge) + " will be charged.")


# %%
# Test serving
import json
import requests

dataset = X_test

ds_dict = dataset.to_dict(orient='split')
del ds_dict["index"]
del ds_dict["columns"]
ds_dict["inputs"] = ds_dict.pop("data")

data_json = json.dumps(ds_dict, allow_nan=True)

url = 'https://adb-5005285748743411.11.azuredatabricks.net/model/LR%20Tech%20Talk%2002/1/invocations'
headers = {'Authorization': f'Bearer {"dapi0fab976aad7153e0e6c40185ac1e4333"}', 'Content-Type': 'application/json'}

response = requests.request(method='POST', headers=headers, url=url, data=data_json)
if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
print(response.json())
