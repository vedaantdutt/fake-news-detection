
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

