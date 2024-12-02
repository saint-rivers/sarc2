import tensorflow as tf
import transformers
import pandas as pd
import data
from keras import backend as K
from sklearn.model_selection import train_test_split
import datapac
import models as modelpac
import utils
from sklearn.metrics import f1_score

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(physical_devices[0])
    tf.config.experimental.set_memory_growth(physical_devices[0], True) 
else:
    print("##### GPU not found #####")
    print("using CPU")

model_name = "cardiffnlp/twitter-roberta-base-sentiment"


### LOADING DATA ###
df = datapac.load_balanced_twitter_reddit_data()
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['sarcastic'], stratify=df['sarcastic'], test_size=0.2, random_state=42)

### DATASET SETUP ###
tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)
train_enc = datapac.tokenize(tokenizer=tokenizer, text=x_train.values.tolist())
test_enc = datapac.tokenize(tokenizer=tokenizer, text=x_test.values.tolist())

train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_enc), y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_enc), y_test))

batch_size = 16
train_dataset = train_dataset.shuffle(len(y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.shuffle(len(y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

### MODEL CHOICE ###
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = modelpac.build_roberta_cnn(model_name=roberta)

### TRAIN ###

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

# model.fit(dataset, epochs=5)
out = model.fit(train_dataset, epochs=5)
results = model.evaluate(test_dataset, batch_size=128)
print(out.history)

import json
with open("results/roberta_cnn.json", "w+") as file:
    file.write(json.dumps(out.history))

model.save("output/sarc_detect_2.keras")
