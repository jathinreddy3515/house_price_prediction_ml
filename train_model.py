import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("Ames_Housing_Subset(in).csv")

# Target variable
y = df["SalePrice"]

# Features
X = df.drop("SalePrice", axis=1)

# Identify categorical & numerical columns
categorical_cols = ["Neighborhood", "Kitchen Qual", "Exter Qual"]
numerical_cols = ["PID", "Year Built", "Overall Qual", "Lot Area"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# Create full pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# Train model
pipeline.fit(X, y)

# Save full pipeline
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model trained and saved successfully.")

