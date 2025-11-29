import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(path="data/netflix.csv"):
    import io
    import pandas as pd

    if isinstance(path, str):
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(io.StringIO(path.getvalue().decode("utf-8")))

    # Strip whitespace in date column
    df['date_added'] = df['date_added'].astype(str).str.strip()

    # Convert date with flexible formatting
    df['date_added'] = pd.to_datetime(
        df['date_added'],
        errors='coerce',
        infer_datetime_format=True
    )

    # Extract year
    df['year_added'] = df['date_added'].dt.year
    df['year_added'] = df['year_added'].fillna(df['year_added'].median())

    # SELECT FEATURES
    X = df[['duration', 'rating', 'listed_in', 'year_added']]
    y = df['type']

    # ENCODE CATEGORICALS
    X['listed_in'] = X['listed_in'].str.split(',').str[0]
    X = pd.get_dummies(X)

    # LABEL ENCODER FOR TARGET
    le = LabelEncoder()
    y = le.fit_transform(y)

    return train_test_split(X, y, test_size=0.2, random_state=42), le
