import pandas as pd
import transformers
# from fairseq_cli.train import train
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

def process_data():
    df = pd.read_csv("data/isarcasm/isarcasm2022.csv")
    df = df[['tweet', 'sarcastic']]
    df.dropna(inplace=True)

    x_train, x_val, y_train, y_val = train_test_split(df['tweet'], df['sarcastic'], test_size=0.2, random_state=34)

    train = pd.DataFrame([x_train, y_train]).T
    train = train.reset_index()[['tweet', 'sarcastic']]
    train.to_csv("data/isarcasm/isarc_train.csv")

    test = pd.DataFrame([x_val, y_val]).T
    test = test.reset_index()[['tweet', 'sarcastic']]
    test.to_csv("data/isarcasm/isarc_test.csv")

def load_headlines():
    train = pd.read_csv('data/huff_onion/headline_train.csv')
    test = pd.read_csv('data/huff_onion/headline_test.csv')

    train_tweets = train['headline'].values.tolist()
    train_labels = train['sarcastic'].values.tolist()
    test_tweets = test['headline'].values.tolist()
    test_labels = test['sarcastic'].values.tolist()

    print(train_tweets)
    return train_tweets, train_labels, test_tweets, test_labels


def load_data_soraby():
  
    
    train = pd.read_csv('data/soraby_sarcasm2/sarc_train.csv')
    test = pd.read_csv('data/soraby_sarcasm2/sarc_test.csv')

    train_tweets = train['text'].values.tolist()
    train_labels = train['class'].values.tolist()
    test_tweets = test['text'].values.tolist()
    test_labels = test['class'].values.tolist()

    return train_tweets, train_labels, test_tweets, test_labels


def load_data_isarc():
    train = pd.read_csv('data/isarcasm/isarc_train.csv')
    test = pd.read_csv('data/isarcasm/isarc_test.csv')

    train_tweets = train['tweet'].values.tolist()
    train_labels = train['sarcastic'].values.tolist()
    test_tweets = test['tweet'].values.tolist()
    test_labels = test['sarcastic'].values.tolist()

    return train_tweets, train_labels, test_tweets, test_labels


def load_data_sarc():
    train = pd.read_csv('data/sarc/Train_Dataset.csv')
    test = pd.read_csv('data/sarc/Test_Dataset.csv')

    train_tweets = train['tweet'].values.tolist()
    train_labels = train['sarcastic'].values.tolist()
    test_tweets = test['tweet'].values.tolist()
    test_labels = test['sarcastic'].values.tolist()

    return train_tweets, train_labels, test_tweets, test_labels


def load_data(dataset: str):
    if dataset == "sarc":
        return load_data_sarc()
    elif dataset == "isarc":
        return load_data_isarc()
    elif dataset == "headlines":
        return load_headlines()
    elif dataset == "soraby":
        return load_data_soraby()
    print("dataset not found")
    return None

def process_soraby(): 
    df = pd.read_csv('data/soraby_sarcasm2/GEN-sarc-notsarc.csv')
    x_train, x_val, y_train, y_val = train_test_split(df['text'], df['class'], test_size=0.2, random_state=34)
    print(x_train.values.tolist())
    
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train).flatten()
    train = pd.DataFrame([x_train, y_train]).T
    train = train.reset_index()[['text', 'class']]
    train.to_csv("data/soraby_sarcasm2/sarc_train.csv")

    y_val = lb.fit_transform(y_val).flatten()
    test = pd.DataFrame([x_val, y_val]).T
    test = test.reset_index()[['text', 'class']]
    test.to_csv("data/soraby_sarcasm2/sarc_test.csv")


def load_train_test_set():
    train_tweets, train_labels, test_tweets, test_labels = load_data("soraby")
    # train_tweets, train_labels, test_tweets, test_labels = load_data("headlines")

    ### START ###
    model_name = "FacebookAI/roberta-base"
    tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)
    # tokenizer, model = TwitterRoberta()

    tokenizer.save_pretrained('output/saved_tokenizer')

    train_encodings = tokenizer(train_tweets, truncation=True, padding=True, return_tensors = 'tf')
    test_encodings = tokenizer(test_tweets, truncation=True, padding=True, return_tensors = 'tf')
    ### END ###

    return train_encodings, test_encodings
    # train_dataset = SarcasmDataset(train_encodings, train_labels)
    # test_dataset = SarcasmDataset(test_encodings, test_labels)

if __name__ == "__main__":
    process_soraby()