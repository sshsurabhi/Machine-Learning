import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df, test_size=0.2, random_state=42):
    """Preprocess data: split into train/test and scale features."""
    # Separate features and target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    df = load_data("data/sample_data.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print("Data preprocessing complete!")