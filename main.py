# %%
import tensorflow as tf
import transformers
import data
from keras import backend as K

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(physical_devices[0])
    tf.config.experimental.set_memory_growth(physical_devices[0], True) 
else:
    print("##### GPU not found #####")
    print("using CPU")
    
# %%
import os
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ['XLA_FLAGS']="/home/dayan/anaconda3/pkgs/cuda-nvvm-tools-12.4.131-h6a678d5_0/nvvm"

# %%
model_name = "FacebookAI/roberta-base"
tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)
x_train, y_train, x_test, y_test = data.load_data("isarc") 

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
    # return {key: encodings[key] for key in encodings}, labels

# %%
train_enc = tokenize_function(x_train)
test_enc = tokenize_function(x_test)

# %%
type(train_enc)

# %%
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_enc), y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_enc), y_test))

# # %%
# dataset.element_spec[0]['input_ids'].

# %%
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

# x1 = tf.keras.layers.Reshape((128 * 128, 1))(x1)
# x1 = tf.keras.layers.Conv1D(filters=128, kernel_size=2,padding='same')(x1)
# x1 = tf.keras.layers.LeakyReLU()(x1)
# x1 = tf.keras.layers.Dense(1)(x1)
# x1 = tf.keras.layers.Flatten()(x1)
# x1 = tf.keras.layers.GlobalMaxPooling1D()(x1)

# x1 = tf.keras.layers.Activation('softmax')(x1)
output = tf.keras.layers.Dense(1, activation="softmax")(x1)

# %%
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
out = model.fit(train_dataset, epochs=1)
results = model.evaluate(test_dataset, batch_size=128)
print(out.history)

import json
with open("results/roberta_cnn.json", "w+") as file:
    file.write(json.dumps(out.history))

model.save("output/sarc_detect.keras")


