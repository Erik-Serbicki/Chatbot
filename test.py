import matplotlib.pyplot as plt
import numpy as np
from utils import *
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import pickle

VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 20



model = tf.keras.models.load_model(
    "model.h5",
    custom_objects={
        "PositionalEncoding": PositionalEncoding,
        "MultiHeadAttentionLayer": MultiHeadAttentionLayer,
    },
    compile=False,
)

from_disk = pickle.load(open("vector_layer.pkl", "rb"))
vectorization_layer = TextVectorization.from_config(from_disk['config'])
vectorization_layer.set_weights(from_disk['weights'])

START_TOKEN= [vectorization_layer.vocabulary_size()]

def evaluate(sentence):
    
    sentence = tf.expand_dims(vectorization_layer(sentence), axis=0)
    output = tf.expand_dims(START_TOKEN, 0)
    
    
    for i in range(MAX_SEQUENCE_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)
        
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        
        if tf.equal(predicted_id, 0):
            break
       
        output = tf.concat([output, predicted_id], axis=-1)
    
    return tf.squeeze(output, axis=0)

def predict(sentence):
    prediction = evaluate(sentence)[1:]
    vocab =  vectorization_layer.get_vocabulary()
    return " ".join([vocab[each] for each in prediction])

def main():
    while True:
        question = input()
        if question == "exit":
            break
        else:
            pred = predict(question)
            print(pred)

main()
        