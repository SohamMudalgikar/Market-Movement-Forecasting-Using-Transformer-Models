# Market-Movement-Forecasting-Using-Transformer-Models


In this project we would try to leverage Transformer-based neural networks to predict the directional movement of stock prices. The model predicts whether the stock price will go up or down in the time t+1 based on prior 2 time steps (from t-2 to t) of historical data segmented data into small intervals (dt). In our approach we focus on practical trading strategies where the direction of price movement is more valuable than the exact price. By focusing on whether the stock price will go up or down in the next time step, we align the model's predictions with practical trading strategies.



## Project Structure


- `python/`: python code for exploratory analysis and development.
  - `data_collection.py`: Collects stock data using yfinance.
  - `feature_engineering.py`: Calculates financial indicators and preprocesses data.
  - `visualize.py`: Visualizes stock data and indicators.
  - `model.py`: Defines the Transformer model architecture.
  - `train.py`: Prepares data sequences and trains the model.
  - `evaluate.py`: Evaluates the model and visualizes results.




