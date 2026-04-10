# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python dependencies
# Note: You'll need to create a requirements.txt from their environment.yml
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly install PyTorch Geometric (PyG) and Scatter (needed for DMPNN)
RUN pip install torch_geometric
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# Copy the rest of the repository, including app.py and your saved model checkpoint
COPY . /app

# Expose port 8080 for Vertex AI
EXPOSE 8080

# Run the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
