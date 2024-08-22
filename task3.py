import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('C:\\Users\\HP\\OneDrive\\Desktop\\Internship\\Task3.csv')
X = df.drop(columns=['species'])
y = df['species']

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

report = classification_report(y_test, y_pred, target_names=le.classes_)
print("\nClassification Report:\n", report)

results_df = X_test.copy()
results_df = pd.DataFrame(results_df, columns=X.columns)
results_df['True Label'] = le.inverse_transform(y_test)
results_df['Predicted Label'] = le.inverse_transform(y_pred)

results_df.to_csv('C:\\Users\\HP\\OneDrive\\Desktop\\Internship\\Task3_Predictions.csv', index=False)
