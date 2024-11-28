# %%
import tensorflow as tf
import transformers
import data

# %%
import os
os.environ['TF_GPU_ALLOCATOR']="cuda_malloc_async"

# %%
model_name = "FacebookAI/roberta-base"
tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)
x_train, y_train, x_test, y_test = data.load_data("soraby") 

# %%
def tokenize_function(texts):
    encodings = tokenizer(
        texts, 
        truncation=True, 
        padding=True, 
        max_length=128, 
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
dataset = tf.data.Dataset.from_tensor_slices((dict(train_enc), y_train))

# # %%
# dataset.element_spec[0]['input_ids'].

# %%
batch_size = 16
dataset = dataset.shuffle(len(y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# %%
config = transformers.BertConfig.from_pretrained(model_name,
                                    output_hidden_states=True)
roberta_model = transformers.TFRobertaModel.from_pretrained(model_name, config=config)
input_ids = tf.keras.Input(shape=(128,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.Input(shape=(128,), dtype=tf.int32, name="attention_mask")
roberta_outputs = roberta_model(input_ids=input_ids, attention_mask=attention_mask)

# print(roberta_outputs)

tf.config.run_functions_eagerly(True)
tf.keras.mixed_precision.set_global_policy("mixed_float16")
tf.debugging.check_numerics(roberta_outputs.last_hidden_state, "Invalid values in last_hidden_state")


# x1 = tf.keras.layers.Dropout(0.1)(roberta_model)
x1 = tf.keras.layers.Conv1D(filters=768, kernel_size=2,padding='same')(roberta_outputs.last_hidden_state)
x1 = tf.keras.layers.LeakyReLU()(x1)
x1 = tf.keras.layers.Dense(1)(x1)
x1 = tf.keras.layers.Flatten()(x1)
output = tf.keras.layers.Activation('softmax')(x1)

# %%
model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

# %%
model.compile(
    optimizer='adam',
    # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    loss='sparse_categorical_crossentropy',
    metrics=["accuracy"]
)

# %%
model.fit(dataset, epochs=3)


