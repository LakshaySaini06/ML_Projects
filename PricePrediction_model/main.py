import os
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"
INPUT_FILE = "input.csv"
OUTPUT_FILE = "output.csv"


# --------------------- BUILD PIPELINE ---------------------
def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

    return full_pipeline


# --------------------- TRAIN MODEL ---------------------
def train_model():
    housing = pd.read_csv("housing.csv")

    # Create income category for stratified sampling
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    # Stratified split
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
        strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

    # Save test data as input.csv for later prediction
    strat_test_set.to_csv(INPUT_FILE, index=False)

    # Separate labels
    housing = strat_train_set.copy()
    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)

    # Build preprocessing pipeline
    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)

    # Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    # Evaluate model
    scores = cross_val_score(model, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=5)
    rmse_scores = np.sqrt(-scores)
    print("\n✅ Model trained successfully!")
    print("Cross-validation RMSE:", rmse_scores.mean())

    # Save artifacts
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print(f"Saved: {MODEL_FILE}, {PIPELINE_FILE}, {INPUT_FILE}")


# --------------------- PREDICT ---------------------
def predict():
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    if not os.path.exists(INPUT_FILE):
        print("⚠️ input.csv not found. Creating a new one...")
        housing = pd.read_csv("housing.csv")
        housing["income_cat"] = pd.cut(
            housing["median_income"],
            bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
            labels=[1, 2, 3, 4, 5]
        )
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)
        strat_test_set.to_csv(INPUT_FILE, index=False)
        print("✅ Created new input.csv")

    input_data = pd.read_csv(INPUT_FILE)
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)

    input_data["predicted_median_house_value"] = predictions
    input_data.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Predictions saved to {OUTPUT_FILE}")


# --------------------- MAIN EXECUTION ---------------------
if not os.path.exists(MODEL_FILE):
    train_model()
else:
    predict()
