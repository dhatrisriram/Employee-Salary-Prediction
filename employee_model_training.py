import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Load and clean data
data = pd.read_csv("adult 3.csv")

# Replace unknowns
data.occupation.replace({'?': 'Others'}, inplace=True)
data.workclass.replace({'?': 'Others'}, inplace=True)
data = data[~data['workclass'].isin(['Without-pay', 'Never-worked'])]
data = data[~data['education'].isin(['5th-6th', '1st-4th', 'Preschool'])]

# Remove outliers
data = data[(data['age'] <= 75) & (data['age'] >= 17)]

# Select only the 5 required columns
features = ['age', 'education', 'occupation', 'hours-per-week']  
target = 'income'

# Encode categorical features
encoders = {}
label_maps = {}
for col in ['education', 'occupation']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le
    label_maps[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# Encode target
target_encoder = LabelEncoder()
data[target] = target_encoder.fit_transform(data[target])

# Prepare input and output
X = data[features]
y = data[target]

# Scale
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train models
models = {
    "KNN": KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(),
    "MLP": MLPClassifier(solver='adam', hidden_layer_sizes=(5,2), random_state=2, max_iter=2000),
    "GradientBoosting": GradientBoostingClassifier()
}

results = {}
for name, model in models.items():
    model.fit(x_train, y_train)
    acc = accuracy_score(y_test, model.predict(x_test))
    results[name] = acc
    print(f"{name}: {acc:.4f}")

# Save best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
joblib.dump(best_model, "best_model.pkl")
print(f"✅ Best model saved: {best_model_name}")

# Save everything else
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")
joblib.dump(label_maps, "label_maps.pkl")
