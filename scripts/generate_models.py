import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from promptena.prompt_classifier import PromptContextClassifier
from sklearn.model_selection import train_test_split


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv("./datasets/promptena_core_dataset.csv")
    # convert boolean value to int
    df["is_context_sufficient"] = df["is_context_sufficient"].astype(int)
    return df


def split_dataset(df, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    return train_df, test_df


def generate_models() -> None:
    model_types = ["logistic_regression", "svm", "decision_tree", "dnn"]

    for model_type in model_types:
        print(f"Training model: {model_type}")
        classifier = PromptContextClassifier(model_type=model_type)
        df = load_dataset()
        train_df, test_df = split_dataset(df)
        X = train_df["prompt"].values
        y = train_df["is_context_sufficient"].values
        classifier.train(X, y)
        if model_type == "dnn":
            classifier.save_model(f"./models/{model_type}.keras")
        else:
            classifier.save_model(f"./models/{model_type}.pkl")
        print(f"Model {model_type} trained and saved.")


if __name__ == "__main__":
    generate_models()
