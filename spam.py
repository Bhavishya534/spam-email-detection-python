# Step 1: import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load dataset
df = pd.read_csv(r"C:\Users\Bhavishya Shetty\Downloads\skill lab\email_spam_dataset.csv", encoding='latin-1')
# Keep only required columns
df = pd.read_csv("email_spam_dataset.csv", encoding='latin-1')

# Clean column names (removes hidden spaces or weird characters)
df.columns = df.columns.str.strip()

# Taking first 2 columns
df = df.iloc[:, :2]

# Renaming columns
df.columns = ['message', 'label']

# Step 3 Convert labels to numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 4: Splitting data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# Step 5: Converting txt to numbers (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 7: Predction
y_pred = model.predict(X_test_vec)
# Step 8: Evaluating Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# Step 9: Test with cstm input
def predict_spam(text):
    text_vec = vectorizer.transform([text])
    result = model.predict(text_vec)
    return "Spam" if result[0] == 1 else "Not Spam"


print("Reached input stage")
msg = input("Enter a message: ")
print("Prediction:", predict_spam(msg))