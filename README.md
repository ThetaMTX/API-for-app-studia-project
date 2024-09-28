# AI Image Classification with FastAPI and PyTorch

## Overview
This project implements an AI image classification model using PyTorch, which can classify images between two classes: Melanoma and Melanocytic Nevi. The model is deployed as a web application using FastAPI, allowing users to upload images for classification.

## Features
- Train a convolutional neural network (CNN) on a dataset of images.
- RESTful API for image classification.
- Confidence scores for predictions.
- User-friendly interface for uploading images.

## Technologies Used
- Python
- PyTorch
- FastAPI
- torchvision
- PIL (Python Imaging Library)

## Getting Started

### Prerequisites
Make sure you have the following installed:
- Python 3.8 or later
- Pip (Python package manager)
- PyTorch with CUDA support (if using a GPU)

### Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```
2. Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. Install the required packages:
```
pip install -r requirements.txt
```
4. Training the Model (if needed)
Prepare your dataset in a directory named dataset with subdirectories for each class (e.g., dataset/Melanoma and dataset/Melanocytic Nevi)[not included because it was 20k files :c].

5. Run the training script:
```
python train.py
```
6. Running the API
Start the FastAPI application:
```
python main.py
```
Open your browser and go to http://127.0.0.1:5000/classify/ to access the API documentation and test the image classification endpoint.
```
API Endpoint
POST /classify/
Description: Classifies an uploaded image.
Request: Requires an image file.
Response:
json
Copy code
{
  "class": "Melanoma",
  "confidence": "92.15%"
}
```
##License
This project is licensed under the MIT License. See the LICENSE file for more details.

##Acknowledgments
Thanks to the PyTorch and FastAPI communities for their resources and support.
