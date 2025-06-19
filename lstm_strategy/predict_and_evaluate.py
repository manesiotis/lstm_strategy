import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, X_test, y_test, scaler=None, inverse_transform=True):
    predictions = model.predict(X_test)

    if inverse_transform and scaler is not None:
        predictions = scaler.inverse_transform(predictions)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    
    return {
        "predictions": predictions.flatten(),
        "actual": y_test.flatten(),
        "mse": mse,
        "mae": mae
    }
