
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


df = pd.read_csv('spam.csv')
df = pd.read_csv('spam.csv', encoding='latin-1')[['label', 'message']]


df.columns = ['label', 'message']


print(df.head())
print(df['label'].value_counts())


df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})


X = df['message']
y = df['label_num']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)


print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))


def predict_spam(message):
    msg_vec = vectorizer.transform([message])
    prediction = model.predict(msg_vec)
    return "Spam" if prediction[0] == 1 else "Ham"

print("\nTest Prediction:", predict_spam("Congratulations! You've won a $1000 Walmart gift card. Call now!"))
print("Test Prediction:", predict_spam("Hey, are we still meeting for lunch tomorrow?"))
