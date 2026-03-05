import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


df = pd.read_csv("support_tickets.csv")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return " ".join(words)

df["cleaned_message"] = df["message"].apply(clean_text)


X = df["cleaned_message"]
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=35
)


nb_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2,max_df=0.9)),
    ("model", MultinomialNB())
])

nb_pipeline.fit(X_train, y_train)
nb_preds = nb_pipeline.predict(X_test)

nb_accuracy = accuracy_score(y_test, nb_preds)

print("Naive Bayes Accuracy:", nb_accuracy)
print(confusion_matrix(y_test, nb_preds))
print(classification_report(y_test, nb_preds))



lr_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2,max_df=0.9)),
    ("model", LogisticRegression(max_iter=1000,class_weight="balanced"))])

lr_pipeline.fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_preds)

print("Logistic Regression Accuracy:", lr_accuracy)
print(confusion_matrix(y_test, lr_preds))
print(classification_report(y_test, lr_preds))



if lr_accuracy > nb_accuracy:
    joblib.dump(lr_pipeline, "ticket_classifier_model.pkl")
    print("Saved Logistic Regression as Best Model")
else:
    joblib.dump(nb_pipeline, "ticket_classifier_model.pkl")
    print("Saved Naive Bayes as Best Model")



def predict_ticket(query):
    query = clean_text(query)
    model = joblib.load("ticket_classifier_model.pkl")
    prediction = model.predict([query])
    return prediction[0]



if __name__ == "__main__":
    user_input = input("Enter Support Query: ")
    print("Predicted Category:", predict_ticket(user_input))