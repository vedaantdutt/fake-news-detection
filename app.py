
import pandas as pd

data = pd.read_csv("dataset/data.csv")
print(data.head())

data = data.dropna()

data = data.sample(frac=1, random_state=42)
print(data.head())

print(data.shape)

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # This removes all non-word characters (e.g., punctuation)
    
    # Tokenize, convert to lowercase, and remove stopwords
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words]
    
    # Join the tokens back into a string and return
    return " ".join(tokens)

data['Body'] = data['Body'].apply(preprocess_text)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(data['Body']).toarray()
y = data['Label']

print(X)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train model
model = RandomForestClassifier()
#model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate on test data
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()


