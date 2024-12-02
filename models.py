import transformers
import tensorflow as tf

def build_roberta_cnn(model_name="FacebookAI/roberta-base"):
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
    x1 = tf.keras.layers.Dense(1, activation="softmax")(x1)
    x1 = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=x1)
    return x1