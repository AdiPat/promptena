import pandas as pd
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
