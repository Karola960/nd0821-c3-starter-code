import sys
from ml.model import compute_model_metrics, inference
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from ml.data import process_data

data = pd.read_csv("../data/census_cleaned.csv")
load_gbc = pickle.load(open("../model/gbclassifier.pkl", "rb"))
encoder = pickle.load(open("../model/encoder.pkl", "rb"))
lb = pickle.load(open("../model/lb.pkl", "rb"))
cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

slice_values = []
feature_value_list = []
precision_value = []
recall_value = []
fbeta_value = []
_, test = train_test_split(data, test_size=0.20)

for cat in cat_features:
    for cls in test[cat].unique():
        df_temp = test[test[cat] == cls]

        X_test, y_test, _, _ = process_data(
            df_temp,
            categorical_features=cat_features,
            label="salary",
            encoder=encoder,
            lb=lb,
            training=False)

        y_preds = inference(load_gbc, X_test)
        prc, rcl, fb = compute_model_metrics(y_test, y_preds)
        metrics = "Category: %s Precision: %s Recall: %s FBeta: %s" % (cat, prc, rcl, fb)
        slice_values.append(metrics)

with open('../model/slice_output.txt', 'w') as out:
    for slice_value in slice_values:
        out.write(slice_value + '\n')
