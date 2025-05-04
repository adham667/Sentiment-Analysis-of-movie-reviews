import os
import pandas as pd

def read_reviews_from_folder(folder_path, label):
    reviews = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    reviews.append({"review": content, "polarity": label})
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return reviews

def main(positive_path, negative_path):
    data = []

    data += read_reviews_from_folder(positive_path, 1)
    data += read_reviews_from_folder(negative_path, 0)

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(df.head(10))
    print(df.info())
    print(df.describe())
    print(df['polarity'].value_counts())
    df.to_csv("reviews.csv", index=False)
    print("Saved to reviews.csv")

if __name__ == "__main__":
    main('./pos', './neg')
