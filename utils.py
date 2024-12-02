import re 
from nltk.corpus import  stopwords


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = text.split()
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return " ".join(filtered_text)

