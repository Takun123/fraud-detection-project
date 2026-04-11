import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data(df: pd.DataFrame):
    """
    Preprocesses the credit card fraud dataset.

    Steps:
        1. Converts Class column from float64 to int.
        2. Scales Amount and Time using StandardScaler fitted ONLY on training data.
        3. Splits data into stratified train/test sets (80/20) to preserve class ratio.

    Args:
        df (pd.DataFrame): Raw deduplicated dataframe with columns
                           [Time, V1-V28, Amount, Class].

    Returns:
        X_train, X_test, y_train, y_test (pd.DataFrame / pd.Series)
    """
    df = df.copy()

    df["Class"] = df["Class"].astype(int)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train[["Amount", "Time"]] = scaler.fit_transform(X_train[["Amount", "Time"]])
    X_test[["Amount", "Time"]] = scaler.transform(X_test[["Amount", "Time"]])

    return X_train, X_test, y_train, y_test