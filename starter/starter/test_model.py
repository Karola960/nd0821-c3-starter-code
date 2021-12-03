import os
import pickle

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model


@pytest.fixture
def root():
    return os.getcwd()


@pytest.fixture
def files(root):
    data = pd.read_csv("../data/census_cleaned.csv")

    model = os.path.join(root, "../model/gbclassifier.pkl")
    with open(model, "rb") as f:
        model = pickle.load(f)

    encoder = os.path.join(root, "../model/encoder.pkl")
    with open(encoder, "rb") as f:
        encoder = pickle.load(f)

    lb = os.path.join(root, "../model/lb.pkl")
    with open(lb, "rb") as f:
        lb = pickle.load(f)

    return data, model, encoder, lb


@pytest.fixture
def train_test_data(files):
    data, model, encoder, lb = files
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    return train, test


def test_train_model(files, root):
    data, model, encoder, lb = files
    train, test = train_test_split(data, test_size=0.20)
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
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    filepath = os.path.join(root, "../model/gbclassifier_test.pkl")
    model = train_model(X_train, y_train, filepath=filepath)

    assert os.path.exists(filepath)
    return X_train, y_train, model, encoder, lb


def test_compute_model_metrics(files, train_test_data):
    _, model, encoder, lb = files
    train, test = train_test_data
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

    X_train, y_train, _, _ = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
        encoder=encoder,
        lb=lb,
    )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    y_train_pred = inference(model, X_train)
    y_test_pred = inference(model, X_test)

    precision_train, recall_train, fbeta_train = compute_model_metrics(
        y_train, y_train_pred
    )
    precision_test, recall_test, fbeta_test = compute_model_metrics(
        y_test, y_test_pred
    )

    assert isinstance(precision_train, float)
    assert isinstance(precision_test, float)
    assert isinstance(recall_train, float)
    assert isinstance(recall_test, float)
    assert isinstance(fbeta_train, float)
    assert isinstance(fbeta_test, float)


def test_inference(train_test_data, files):
    _, model, encoder, lb = files
    train, test = train_test_data

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
    X_train, y_train, _, _ = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
        encoder=encoder,
        lb=lb,
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    y_train_pred = inference(model, X_train)
    assert len(y_train_pred) == X_train.shape[0]
    assert len(y_train_pred) > 0

    y_test_pred = inference(model, X_test)
    assert len(y_test_pred) == X_test.shape[0]
    assert len(y_test_pred) > 0
