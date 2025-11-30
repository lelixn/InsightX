import pandas as pd
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(file):

    # Read CSV safely
    if isinstance(file, str):
        df = pd.read_csv(file)
    else:
        df = pd.read_csv(io.StringIO(file.getvalue().decode("utf-8")))

    df = df.copy()

    # -------------------------
    # 1. Handle date columns
    # -------------------------
    if 'date_added' in df.columns:
        df['date_added'] = df['date_added'].astype(str).str.strip()
        df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
        df['year_added'] = df['date_added'].dt.year
        df['year_added'] = df['year_added'].fillna(df['year_added'].median())
    else:
        # If no date column, create a dummy year
        df['year_added'] = 2000

    # -------------------------
    # 2. Handle duration column
    # -------------------------
    if 'duration' in df.columns:
        def clean_duration(value):
            value = str(value)
            if "min" in value:
                return int(value.replace(" min", "").strip())
            else:
                try:
                    seasons = int(value.split()[0])
                    return seasons * 60
                except:
                    return 60
        df['duration_cleaned'] = df['duration'].apply(clean_duration)
    else:
        df['duration_cleaned'] = 60

    # -------------------------
    # 3. Handle rating column
    # -------------------------
    if 'rating' in df.columns:
        df['rating'] = df['rating'].astype(str)
    else:
        df['rating'] = "Unknown"

    # -------------------------
    # 4. Handle genre column
    # -------------------------
    if 'listed_in' in df.columns:
        df['listed_in'] = df['listed_in'].astype(str).str.split(',').str[0]
    else:
        df['listed_in'] = "Other"

    
    if 'type' not in df.columns:
        # Fail safely and inform Streamlit UI
        raise ValueError("❌ Dataset must contain a 'type' column to train ML models.")

    y = df['type']

    # -------------------------
    # 6. Build feature matrix
    # -------------------------
    X = df[['duration_cleaned', 'rating', 'listed_in', 'year_added']]
    X = pd.get_dummies(X)

    # -------------------------
    # 7. Encode target
    # -------------------------
    le = LabelEncoder()
    y = le.fit_transform(y)

    # -------------------------
    # 8. Split data
    # -------------------------
    return train_test_split(X, y, test_size=0.2, random_state=42), le
def get_models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Support Vector Machine": SVC(probability=True)
    }
    return models