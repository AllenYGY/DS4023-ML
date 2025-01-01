import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = "basketball.csv"
df = pd.read_csv(file_path)

# Preprocess the data: Convert categorical columns to numeric using pd.get_dummies
df_encoded = pd.get_dummies(df.drop('play', axis=1))  # Dropping target column 'play' for encoding features
y = df['play'].apply(lambda x: 1 if x == 'yes' else 0)  # Convert 'yes'/'no' to 1/0 for target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=0.3, random_state=42)

# Create and train Naive Bayes classifier
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Make predictions
y_pred = nb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")