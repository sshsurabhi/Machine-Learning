from flask import Flask, request, jsonify
import torch
import numpy as np
from scripts.train import SimpleNN

app = Flask(__name__)

# Load the trained model
model = SimpleNN(input_size=4, hidden_size=10, output_size=1)
model.load_state_dict(torch.load("models/model.pth"))
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["data"]
    data = np.array(data, dtype=np.float32)
    data = torch.tensor(data, dtype=torch.float32)
    with torch.no_grad():
        output = model(data)
        prediction = (output > 0.5).float().item()
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)