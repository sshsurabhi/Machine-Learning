import torch
from scripts.preprocess import load_data, preprocess_data
from scripts.train import SimpleNN

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test set."""
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    with torch.no_grad():
        outputs = model(X_test)
        predicted = (outputs > 0.5).float()
        accuracy = (predicted == y_test).float().mean()

    print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")

if __name__ == "__main__":
    # Load and preprocess data
    df = load_data("data/sample_data.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Load the trained model
    input_size = X_train.shape[1]
    model = SimpleNN(input_size, hidden_size=10, output_size=1)
    model.load_state_dict(torch.load("models/model.pth"))

    # Evaluate the model
    evaluate_model(model, X_test, y_test)