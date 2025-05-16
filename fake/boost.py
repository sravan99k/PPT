import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the CSV
import pandas as pd

df = pd.read_csv("Student_Survey_Responses_300.csv", encoding='latin1')

print("Columns in the dataset:")
print(df.columns.tolist())

print("\nSample data:")
print(df.head())


# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Check required column exists
if "Mental Health Status" not in df.columns:
    raise ValueError("Mental Health Status column not found. Available columns: " + str(df.columns))

# Encode the target column
df["Mental Health Category Encoded"] = df["Mental Health Status"].astype("category").cat.codes

# Drop unnecessary columns (Name, Timestamp, Mental Health Status)
df = df.drop(["Timestamp", "Name", "Mental Health Status"], axis=1)

# Encode all categorical columns
label_encoders = {}
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Features and target
X = df.drop("Mental Health Category Encoded", axis=1)
y = df["Mental Health Category Encoded"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))

# Save model and encoders
joblib.dump(model, "mental_health_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("\n✅ Model and encoders saved successfully.")
