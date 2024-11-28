import tensorflow as tf
import transformers

import data


model_name = "FacebookAI/roberta-base"
tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)
# out = tokenizer("Hello world")
# input_ids = out['input_ids']
# attention_mask = out['attention_mask']
# train, test = data.load_train_test_set()
# type(train)
#

x_train, x_label, _, _ = data.load_data("soraby") 
def tokenize_function(texts, labels):
    encodings = tokenizer(
        texts, 
        truncation=True, 
        padding=True, 
        max_length=128, 
        return_tensors="tf"
    )
    return {key: encodings[key] for key in encodings}, labels

encodings, labels = tokenize_function(x_train, x_label)
dataset = tf.data.Dataset.from_tensor_slices((encodings, labels))
batch_size = 8
batched_dataset = dataset.batch(batch_size)

# optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


bert_model = transformers.TFRobertaModel.from_pretrained(model_name, num_labels=2)
# x = bert_model()
for batch in batched_dataset:
    inputs, batch_labels = batch
    # with tf.GradientTape() as tape:
    outputs = bert_model(batch, training=True)

# x1 = tf.keras.layers.Dropout(0.1)(x[0])
# x1 = tf.keras.layers.Conv1D(768, 2,padding='same')(x1)
# x1 = tf.keras.layers.LeakyReLU()(x1)
# x1 = tf.keras.layers.Dense(1)(x1)
# x1 = tf.keras.layers.Flatten()(x1)
# x1 = tf.keras.layers.Activation('softmax')(x1)
