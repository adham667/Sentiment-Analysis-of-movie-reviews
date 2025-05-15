import os
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


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

    return " ".join(tokens)


def main(positive_path, negative_path):
    data = []

    data += read_reviews_from_folder(positive_path, 1)
    data += read_reviews_from_folder(negative_path, 0)

    df = pd.DataFrame(data)
    df["review"] = df["review"].apply(clean_text)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(df.head(10))
    print(df.info())
    print(df.describe())
    print(df["polarity"].value_counts())
    df.to_csv("reviews.csv", index=False)
    print("Saved to reviews.csv")

    # Class distribution
    df["polarity"].value_counts().plot(kind="bar", title="Class Distribution")
    plt.xlabel("Polarity")
    plt.ylabel("Count")
    plt.show()

    # Word cloud for all reviews
    text_all = " ".join(df["review"])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_all)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Reviews")
    plt.show()

    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(df["review"])
    print("TF-IDF matrix shape:", X_tfidf.shape)

    y = df["polarity"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print("\nLogistic Regression Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr))

    # 2. Multinomial Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    print("\nMultinomial Naive Bayes Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_nb))
    print(classification_report(y_test, y_pred_nb))

    # 3. Support Vector Machine (LinearSVC)
    svm = LinearSVC()
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    print("\nSupport Vector Machine (LinearSVC) Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_svm))

    # Model accuracy comparison
    accuracies = [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_nb),
        accuracy_score(y_test, y_pred_svm)
    ]
    models = ["Logistic Regression", "Naive Bayes", "LinearSVC"]
    plt.bar(models, accuracies, color=["skyblue", "lightgreen", "salmon"])
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.ylim(0, 1)
    plt.show()


if __name__ == "__main__":
    main("./review_polarity/pos", "./review_polarity/neg")
