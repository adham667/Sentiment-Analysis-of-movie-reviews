import os
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer  # Add WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV  # Add GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import VotingClassifier  # Add VotingClassifier
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import joblib
import spacy

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def read_reviews_from_folder(folder_path, label):
    reviews = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    reviews.append({"review": content, "polarity": label})
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return reviews


stop_words = set(stopwords.words("english")) - {"not", "no"}
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()  # Initialize lemmatizer


def clean_text(text):
    # Remove HTML
    text = re.sub(r"<.*?>", " ", text)

    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()

    # remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]

    # stemming the text
    tokens = [stemmer.stem(word) for word in tokens]
    # lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)

def pos_filter(text):
    doc = nlp(text)
    return " ".join(tok.text for tok in doc if tok.pos_ in {"ADJ", "VERB", "NOUN", "ADV"})

def main(positive_path, negative_path):
    data = []

    data += read_reviews_from_folder(positive_path, 1)
    data += read_reviews_from_folder(negative_path, 0)

    df = pd.DataFrame(data)
    print(f"Total reviews: {len(df)}")
    # Apply POS filtering before cleaning
    df["review"] = df["review"].apply(pos_filter)

    
    df["review"] = df["review"].apply(clean_text)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Word cloud for all reviews
    text_all = " ".join(df["review"])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_all)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Reviews")
    plt.show()

    
    vectorizer = TfidfVectorizer()
    X_train_text, X_test_text, y_train, y_test = train_test_split(df["review"], df["polarity"], test_size=0.2, random_state=42)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test  = vectorizer.transform(X_test_text)


    # 1. Logistic Regression with best params (example: C=1, penalty='l2', solver='liblinear')
    lr = LogisticRegression(C=1, penalty='l2', solver='liblinear', max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print("\nLogistic Regression Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr))
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    ConfusionMatrixDisplay(cm_lr, display_labels=["Negative", "Positive"]).plot(cmap="Blues")
    plt.title("Logistic Regression Confusion Matrix")
    plt.show()

    # 2. Multinomial Naive Bayes with best params (example: alpha=0.5)
    nb = MultinomialNB(alpha=0.5)
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    print("\nMultinomial Naive Bayes Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_nb))
    print(classification_report(y_test, y_pred_nb))
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    ConfusionMatrixDisplay(cm_nb, display_labels=["Negative", "Positive"]).plot(cmap="Greens")
    plt.title("Naive Bayes Confusion Matrix")
    plt.show()

    # 3. LinearSVC with best params (example: C=1)
    svm = LinearSVC(C=1)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    print("\nSupport Vector Machine (LinearSVC) Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_svm))
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    ConfusionMatrixDisplay(cm_svm, display_labels=["Negative", "Positive"]).plot(cmap="Oranges")
    plt.title("LinearSVC Confusion Matrix")
    plt.show()

    # Voting Ensemble using best estimators
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', lr),
            ('nb', nb),
            ('svm', svm)
        ],
        voting='hard'
    )
    voting_clf.fit(X_train, y_train)
    y_pred_voting = voting_clf.predict(X_test)
    print("\nVoting Ensemble Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_voting))
    print(classification_report(y_test, y_pred_voting))
    cm_voting = confusion_matrix(y_test, y_pred_voting)
    ConfusionMatrixDisplay(cm_voting, display_labels=["Negative", "Positive"]).plot(cmap="Purples")
    plt.title("Voting Ensemble Confusion Matrix")
    plt.show()

    # Model accuracy comparison
    accuracies = [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_nb),
        accuracy_score(y_test, y_pred_svm),
        accuracy_score(y_test, y_pred_voting)
    ]
    models = ["Logistic Regression", "Naive Bayes", "LinearSVC", "Voting Ensemble"]
    plt.bar(models, accuracies, color=["skyblue", "lightgreen", "salmon", "purple"])
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.ylim(0, 1)
    plt.show()

    # Save the best model
    best_index = accuracies.index(max(accuracies))
    best_model = [lr, nb, svm, voting_clf][best_index]
    joblib.dump(best_model, "best_model.joblib")
    best_model_name = models[best_index]
    print(f"\nBest model '{models[best_index]}' saved as '{best_model_name}'")


if __name__ == "__main__":
    main("./review_polarity/pos", "./review_polarity/neg")
