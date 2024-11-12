# Visualize the results

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score
# from train import custom_mae_loss, dir_accuracy

if __name__ == "__main__":
    data = pd.read_csv("data/processed_stock_data.csv", index_col=0, parse_dates=True)
    seq_length = 24
    sequences, labels = create_sequences(data, seq_length)
    _, X_test, _, y_test = train_test_split(sequences, labels, test_size=0.1, random_state=42)
    model = load_model('models/transformer_model.keras', custom_objects={'custom_mae_loss': custom_mae_loss, 'dir_accuracy': dir_accuracy})
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    print(f"R-squared Score: {r2}")

    # Plot true vs predicted prices
    plt.figure(figsize=(14, 7))
    plt.plot(y_test, label='True Prices')
    plt.plot(predictions, label='Predicted Prices', alpha=0.7)
    plt.title('True vs Predicted Stock Prices')
    plt.xlabel('Samples')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.savefig('visualizations/true_vs_predicted_prices.png')
    plt.show()
