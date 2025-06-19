# README.md

"""
# Project 9: LSTM Strategy — Predicting Stock Prices with Deep Learning

This project implements a deep learning strategy using Long Short-Term Memory (LSTM) neural networks to forecast stock prices. The model is trained on historical closing prices and evaluated using standard regression metrics.

---

## 📁 Project Structure

lstm_strategy/
├── data_loader.py               # Load stock price data
├── preprocessing.py             # Data scaling and sequence creation
├── model.py                     # LSTM model architecture
├── predict_and_evaluate.py     # Generate predictions and compute MSE/MAE
├── plotting.py                 # Plot actual vs predicted prices
├── run_lstm.py                  # Main pipeline script
├── notebook.ipynb              # Optional Jupyter notebook
├── requirements.txt
├── README.md
└── plots/                      # Output directory for generated plots

---

## 🚀 How to Run

1. Clone the repository:

    git clone https://github.com/yourusername/lstm_strategy.git
    cd lstm_strategy

2. (Optional) Create and activate a virtual environment:

    python -m venv venv
    source venv/bin/activate       # On Windows: .\\venv\\Scripts\\activate

3. Install dependencies:

    pip install -r requirements.txt

4. Run the main script:

    python run_lstm.py

---

## 📈 Output

- Prediction plots saved in the `plots/` directory
- Evaluation metrics (MSE, MAE) printed to the terminal

---

## 🔬 Methodology

- LSTM model with one hidden layer and dropout
- Scaled time-series data with lookback window
- Evaluation on hold-out test set
- Visual comparison of predictions vs actual prices

---

## 🧪 Optional

Use `notebook.ipynb` for experimentation and fine-tuning inside a Jupyter environment.

---

## 📦 Dependencies

See `requirements.txt`. Includes:
- yfinance
- pandas
- numpy
- scikit-learn
- matplotlib
- tensorflow

---

## 📬 Contact

For suggestions or questions, open an issue or pull request.
"""
