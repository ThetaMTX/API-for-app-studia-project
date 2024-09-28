from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms
import torch.nn.functional as F
import io
from fastapi.middleware.cors import CORSMiddleware
import logging

from model import SimpleCNN

# Set up logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to allow specific origins if necessary
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = SimpleCNN(num_classes=2)
model.load_state_dict(torch.load('modelreal.pth'))  # Ensure the path is correct
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Define recognized classes
recognized_classes = {0: "Melanoma", 1: "Melanocytic Nevi"}

@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    try:
        # Attempt to open the image
        image = Image.open(io.BytesIO(await file.read()))

        # Apply transformations and add a batch dimension
        image = transform(image).unsqueeze(0)

        # Run the model inference
        with torch.no_grad():
            output = model(image)
            probabilities = F.softmax(output, dim=1)  # Apply softmax to get probabilities
            confidence, predicted = torch.max(probabilities, 1)  # Get the confidence score and predicted class

        predicted_class = predicted.item()
        confidence_score = confidence.item() * 100  # Convert confidence to percentage

        # Check if the predicted class is valid
        if predicted_class in recognized_classes:
            return {
                "class": recognized_classes[predicted_class],
                "confidence": f"{confidence_score:.2f}%"  # Return confidence score
            }
        else:
            raise HTTPException(status_code=400, detail="The model does not recognize the image.")

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")  # Log the error
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5000)  # Use localhost for local testing
