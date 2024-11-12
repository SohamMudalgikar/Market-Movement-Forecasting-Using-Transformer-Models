#  Model Training

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split
from model import build_model


def create_sequences(data, seq_length):
    # Select only numeric features for the sequences
    features = ['Close', 'BB_width', 'RSI', 'ROC', 'Volume']  # List your numeric features here
    sequences = []
    labels = []
    for i in range(len(data) - seq_length - 12):
        # Use .loc to select specific columns and rows
        seq = data.loc[data.index[i:i+seq_length], features].values
        label = data['Close'].iloc[i+seq_length+12]  
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

def custom_mae_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def dir_accuracy(y_true, y_pred):
    return tf.reduce_mean(tf.cast(tf.equal(tf.sign(y_true - y_true), tf.sign(y_pred - y_true)), tf.float32))

if __name__ == "__main__":
    data = pd.read_csv("data/processed_stock_data.csv", index_col=0, parse_dates=True)
    seq_length = 24  
    sequences, labels = create_sequences(data, seq_length)
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.1, random_state=42)
    model = build_model(
        input_shape=X_train.shape[1:],
        head_size=256,
        num_heads=4,
        ff_dim=256,
        num_layers=4,
        dropout=0.1
    )
    model.compile(
        loss=custom_mae_loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=[dir_accuracy]
    )
    checkpoint = ModelCheckpoint('models/transformer_model.keras', save_best_only=True, monitor='val_loss', mode='min')
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=64,
        callbacks=[checkpoint]
    )
