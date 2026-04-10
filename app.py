from fastapi import FastAPI, Request
import torch
import uvicorn
from models import MolSets # or whatever model class you trained (e.g., from dmpnn)
# import data_utils ... you will need to import your data processing logic here

app = FastAPI()

# 1. Load the model at startup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = "results/your_model_checkpoint.pt" # Update with your trained weights

# Initialize your model architecture (make sure hyperparameters match training)
model = MolSets(...).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(request: Request):
    # Vertex AI sends requests in a JSON format {"instances": [...]}
    body = await request.json()
    instances = body["instances"]
    
    predictions = []
    for instance in instances:
        # 1. Extract inputs (e.g., SMILES, solvent weight fractions, salt molality)
        # 2. Convert raw inputs to graph format using MolSets' data_utils.py
        # graph_data = data_utils.process_my_data(...) 
        
        # 3. Run inference
        with torch.no_grad():
            output = model(graph_data)
            predictions.append(output.item())
            
    return {"predictions": predictions}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
