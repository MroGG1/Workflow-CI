import pandas as pd
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Aktifkan Autologging - Ini akan otomatis mencatat semuanya ke Run yang sedang berjalan
mlflow.sklearn.autolog()

def train_model():
    # Load data
    data_path = "namadataset_preprocessing/diabetes_cleaned.csv"
    df = pd.read_csv(data_path)

    # Pisahkan fitur dan target
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Latih Model (Autologging akan otomatis mencatat parameter dan metrik di sini)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediksi & Evaluasi
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Model berhasil dilatih dengan Akurasi: {acc:.4f}")

if __name__ == "__main__":
    train_model()
