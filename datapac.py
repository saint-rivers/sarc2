import pandas as pd
from utils import remove_stopwords, remove_urls

def load_balanced_twitter_reddit_data():
    twitter_train = pd.read_csv('data/Mutant Datasets/dataset_m_111.csv')
    twitter_train['tweet'] = twitter_train['tweet'].apply(remove_stopwords)
    twitter_train['tweet'] = twitter_train['tweet'].apply(remove_urls)
    twitter_train['text'] = twitter_train['tweet']
    twitter_train['source'] = "twitter"
    twitter_train = twitter_train[['text', 'sarcastic', 'source']]

    reddit_train = pd.read_csv('data/soraby_sarcasm2/sarc_train.csv')
    reddit_train['text'] = reddit_train['text'].apply(remove_stopwords)
    reddit_train['text'] = reddit_train['text'].apply(remove_urls)
    reddit_train['sarcastic'] = reddit_train['class']
    reddit_train['source'] = "reddit"
    reddit_train = reddit_train[['text', 'sarcastic', 'source']]

    df = pd.concat([reddit_train, twitter_train],axis=0, ignore_index=True)

    from sklearn.utils import resample

    minority_class = df[df['sarcastic'] == 1]
    majority_class = df[df['sarcastic'] == 0]
    minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
    df = pd.concat([majority_class, minority_upsampled])
    return df


def tokenize(tokenizer, text):
    return tokenizer(
        text, 
        max_length=128, 
        padding="max_length", 
        truncation=True, 
        return_tensors="tf"
    )