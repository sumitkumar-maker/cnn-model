import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

# ✅ CNN model (same as training)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(64 * 5 * 5, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 64 * 5 * 5)
        x = self.fc(x)
        return x

# ✅ Load Model
model = CNN()
model.load_state_dict(torch.load("mnist_model.pth", map_location="cpu"))
model.eval()

# ✅ Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

st.title("🧠 MNIST Digit Classifier (CNN)")
st.write("Upload a digit image (0–9)")

file = st.file_uploader("Upload image", type=["png","jpg","jpeg"])

if file:
    img = Image.open(file).convert("L")

    # ✅ invert image (very important)
    img = TF.invert(img)

    st.image(img, caption="Uploaded Image", width=200)

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    st.success(f"✅ Predicted Digit: {predicted.item()}")
