import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.compose import  ColumnTransformer

data = pd.read_csv("dataset.csv")
data.columns = data.columns.str.strip()

category = ["State","Crop"]
values = ["CostCultivation","Production","Yield","Temperature","RainFall"]

X = data.drop(["Price","CostCultivation2"],axis=1)
y = data["Price"]

ct  = ColumnTransformer([
    ("encoder",OneHotEncoder(),category),
    ("scaler",StandardScaler(),values)
])
model = Pipeline(
    [
        ("ct",ct),
        ("clf",Ridge(max_iter=1000,alpha=1e-6))
    ]
)
model.fit(X,y)

yp = model.predict(X)
print(r2_score(y,yp))
print(mean_squared_error(y,yp))