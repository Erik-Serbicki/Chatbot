from utils import *
import pandas as pd
from datetime import datetime
import emoji
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import utils
from tensorflow.keras.layers import TextVectorization


# Read in the csv file
df = pd.read_csv("jonchats.csv", header=0)

# Format the data
df = format_data(df)
dropIndex = df[(df['date_num'] == 20220511) & (df['sender'] == 2)].index
df.drop(dropIndex, inplace=True)
df.drop(df.tail(43).index, inplace=True)

X = []
Y = []

for i in range(len(df)-1):
    if (df['sender'][i] == 1) & (df['sender'][i+1]==2):
        X.append(df['msg'][i])
        Y.append(df['msg'][i+1])
    
print(len(X))

# Tokenize and Vectorize data
NUM_LAYERS = 3
VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 30
D_MODEL = 256
NUM_HEADS = 4
UNITS = 256
DROPOUT = 0.05

EPOCHS = 30

int_vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH)
int_vectorize_layer.adapt(df['msg'])

pickle.dump({
    'config': int_vectorize_layer.get_config(),
    'weights': int_vectorize_layer.get_weights()
}, open("vector_layer.pkl", "wb"))

x_train = int_vectorize_layer(X)
y_train = int_vectorize_layer(Y)

# Create tensorflow dataset with tokenized data
train_ds = tf.data.Dataset.from_tensor_slices(({"inputs":x_train, "dec_inputs":y_train[:, :-1]}, y_train[:, 1:]))

# For tf.data.Dataset
BATCH_SIZE = 32
BUFFER_SIZE = 100

train_ds = train_ds.cache()
train_ds = train_ds.shuffle(BUFFER_SIZE)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

# Define loss function the model
def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_SEQUENCE_LENGTH - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)

# Clear backend
tf.keras.backend.clear_session()

# Learning rate and optimizer
learning_rate = CustomSchedule(D_MODEL)
optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
)

# Accuracy function
def accuracy(y_true, y_pred):
    # ensure labels have shape (batch_size, MAX_LENGTH - 1)
    y_true = tf.reshape(y_true, shape=(-1, MAX_SEQUENCE_LENGTH - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


strategy = tf.distribute.get_strategy()
with strategy.scope():
    model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    )
    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

model.summary()

model.fit(train_ds, epochs=EPOCHS)


model.save("model.h5")

del model
tf.keras.backend.clear_session()