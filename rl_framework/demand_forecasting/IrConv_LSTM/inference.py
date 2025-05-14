import torch
import numpy as np
from torch.autograd import Variable


def perform_inference(model_path, closeness_data_np, period_data_np, trend_data_np):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()  # Set the model to evaluation mode

    # Convert numpy arrays to PyTorch tensors
    # Add batch dimension (unsqueeze(0))
    # Expected shape for model input: (batch_size, seq_len, num_nodes, features_per_node)
    closeness_tensor = Variable(torch.FloatTensor(closeness_data_np).unsqueeze(0)).to(
        device
    )
    period_tensor = Variable(torch.FloatTensor(period_data_np).unsqueeze(0)).to(device)
    trend_tensor = Variable(torch.FloatTensor(trend_data_np).unsqueeze(0)).to(device)

    with torch.no_grad():  # Disable gradient calculations for inference
        prediction_tensor = model(closeness_tensor, period_tensor, trend_tensor)

    # Model output is typically (batch_size, 1, num_nodes, 1)
    # Squeeze to get (num_nodes,) for a single prediction item
    prediction_np = prediction_tensor.squeeze().cpu().numpy()

    return prediction_np


if __name__ == "__main__":
    # --- Configuration for Inference ---
    # Path to trained model
    MODEL_PATH = "/home/ruroit00/rebalancing_framework/rl_framework/demand_forecasting/models/irregular_convolution_LSTM_35_1747051152_dropoff.pkl"

    NUM_NODES = 236  # number of grid cells
    FEATURES_PER_NODE = 1

    CLOSENESS_SEQ_LEN = 24
    PERIOD_SEQ_LEN = 7
    TREND_SEQ_LEN = 2

    print(f"Attempting to load model from: {MODEL_PATH}")
    print(
        f"Expected input params: NUM_NODES={NUM_NODES}, FEATURES_PER_NODE={FEATURES_PER_NODE}"
    )
    print(
        f"CLOSENESS_SEQ_LEN={CLOSENESS_SEQ_LEN}, PERIOD_SEQ_LEN={PERIOD_SEQ_LEN}, TREND_SEQ_LEN={TREND_SEQ_LEN}"
    )

    # Create dummy input data for testing
    dummy_closeness_data = np.random.rand(
        CLOSENESS_SEQ_LEN, NUM_NODES, FEATURES_PER_NODE
    ).astype(np.float32)
    dummy_period_data = np.random.rand(
        PERIOD_SEQ_LEN, NUM_NODES, FEATURES_PER_NODE
    ).astype(np.float32)
    dummy_trend_data = np.random.rand(
        TREND_SEQ_LEN, NUM_NODES, FEATURES_PER_NODE
    ).astype(np.float32)

    print(f"\nShape of dummy closeness data: {dummy_closeness_data.shape}")
    print(f"Shape of dummy period data: {dummy_period_data.shape}")
    print(f"Shape of dummy trend data: {dummy_trend_data.shape}")

    try:
        prediction = perform_inference(
            MODEL_PATH, dummy_closeness_data, dummy_period_data, dummy_trend_data
        )
        print("\nInference successful.")
        print(f"Prediction shape: {prediction.shape}")
        print(f"First 5 predicted values: {prediction[:5]}")

    except Exception as e:
        print(f"\nAn error occurred during inference: {e}")
        print("Ensure the model file is compatible and input data shapes are correct.")
        print(
            "If the error mentions missing classes (like Irregular_Convolution_LSTM), "
            "ensure the model's class definition is available in the Python environment."
        )
