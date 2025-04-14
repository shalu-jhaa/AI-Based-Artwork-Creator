from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)

# Load the pre-trained model (use your model here)
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    image = Image.open(file.stream)
    image = transform(image).unsqueeze(0)

    # Predict using the model (replace with your model's logic)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    # Return the prediction result
    return jsonify({'prediction': str(predicted.item())})

if __name__ == '__main__':
    app.run(debug=True)

