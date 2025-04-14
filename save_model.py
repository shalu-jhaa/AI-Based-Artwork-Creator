import torch

# Load the pre-trained ResNet-18 model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

# Save the model's state dictionary
torch.save(model.state_dict(), 'resnet18.pth')
