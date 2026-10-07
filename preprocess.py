import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9+\.]", " ", text)     # keep letters, numbers, plus, dot
    tokens = text.split()

    filtered = [
        LEMMATIZER.lemmatize(t)
        for t in tokens
        if t not in STOP_WORDS and len(t) > 2
    ]

    return " ".join(filtered)
