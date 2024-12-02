# %%
import tensorflow as tf
import transformers
import data
from keras import backend as K
import pandas as pd
import re
from nltk.corpus import  stopwords

# %%
# model_name = "FacebookAI/roberta-base"
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)

# %%
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

# %%
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = text.split()
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return " ".join(filtered_text)

# twitter_train = pd.read_csv('data/Mutant Datasets/dataset_m_111.csv')
# twitter_train['tweet'] = twitter_train['tweet'].apply(remove_stopwords)
# twitter_train['tweet'] = twitter_train['tweet'].apply(remove_urls)
# twitter_train['text'] = twitter_train['tweet']
# twitter_train['source'] = "twitter"
# twitter_train = twitter_train[['text', 'sarcastic', 'source']]

# reddit_train = pd.read_csv('data/soraby_sarcasm2/sarc_train.csv')
# reddit_train['text'] = reddit_train['text'].apply(remove_stopwords)
# reddit_train['text'] = reddit_train['text'].apply(remove_urls)
# reddit_train['sarcastic'] = reddit_train['class']
# reddit_train['source'] = "reddit"
# reddit_train = reddit_train[['text', 'sarcastic', 'source']]

# df = pd.concat([reddit_train, twitter_train],axis=0, ignore_index=True)
df = pd.read_csv('data/mydata/mutated_twitter_reddit.csv')

# %%
from sklearn.utils import resample

minority_class = df[df['sarcastic'] == 1]
majority_class = df[df['sarcastic'] == 0]
minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)

# %%
df = pd.concat([majority_class, minority_upsampled])


# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['sarcastic'], stratify=df['sarcastic'], test_size=0.2, random_state=42)

# %%
def tokenize_function(texts):
    encodings = tokenizer(
        texts, 
        max_length=128, 
        padding="max_length", 
        truncation=True, 
        return_tensors="tf"
    )
    return encodings

# %%
train_enc = tokenize_function(x_train.astype(str).values.tolist())
test_enc = tokenize_function(x_test.astype(str).values.tolist())

train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_enc), y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_enc), y_test))

batch_size = 16
train_dataset = train_dataset.shuffle(len(y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.shuffle(len(y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# %%
config = transformers.BertConfig.from_pretrained(model_name,
                                    output_hidden_states=True)
roberta_model = transformers.TFRobertaModel.from_pretrained(model_name, config=config)
input_ids = tf.keras.Input(shape=(128,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.Input(shape=(128,), dtype=tf.int32, name="attention_mask")
roberta_outputs = roberta_model(input_ids=input_ids, attention_mask=attention_mask)

x1 = tf.keras.layers.Dropout(0.1)(roberta_outputs.last_hidden_state)
x1 = tf.keras.layers.Conv1D(filters=128, kernel_size=3,padding='same')(x1)
x1 = tf.keras.layers.LeakyReLU()(x1)
x1 = tf.keras.layers.GlobalMaxPooling1D()(x1)
output = tf.keras.layers.Dense(1, activation="softmax")(x1)
model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
print(model.summary())

# %%
def f1_score(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Threshold predictions
    tp = K.sum(y_true * y_pred)  # True positives
    fp = K.sum((1 - y_true) * y_pred)  # False positives
    fn = K.sum(y_true * (1 - y_pred))  # False negatives

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-8),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(), 
        tf.keras.metrics.FalseNegatives(),
        tf.keras.metrics.FalsePositives(),
        tf.keras.metrics.BinaryCrossentropy(),
        f1_score
    ]
)


# %%
# model.fit(dataset, epochs=5)
out = model.fit(train_dataset, epochs=20)
results = model.evaluate(test_dataset, batch_size=128)
print(out.history)
print("eval")
print(results)

import json
with open("results/roberta_cnn.json", "w+") as file:
    file.write(json.dumps(out.history))

model.save("output/sarc_detect_2.keras")


