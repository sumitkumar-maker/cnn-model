import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
import numpy as np

# -------------------------
# ✅ CNN Model Definition
# -------------------------
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

# -------------------------
# ✅ Load the trained model
# -------------------------
model = CNN()
model.load_state_dict(torch.load("mnist_model.pth", map_location="cpu"))
model.eval()

# -------------------------
# ✅ MNIST Transform
# -------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -------------------------
# ✅ Streamlit UI
# -------------------------
st.title("🧠 MNIST Digit Classifier (Advanced Preprocessing)")
st.write("Upload a handwritten digit image taken from your phone or scanned paper.")

uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

if uploaded:
    # Load & show original
    original_img = Image.open(uploaded).convert("L")
    st.image(original_img, caption="Original Uploaded Image", width=200)

    # -------------------------
    # ✅ STEP 1: Invert Image (phone images are black on white)
    # -------------------------
    img = ImageOps.invert(original_img)

    # -------------------------
    # ✅ STEP 2: Convert to numpy & threshold to remove noise
    # -------------------------
    img_np = np.array(img)
    binary = img_np < 200   # True where digit is

    # If no digit detected
    if not binary.any():
        st.error("Digit not detected. Try clearer photo.")
    else:
        # -------------------------
        # ✅ STEP 3: Find bounding box around the digit
        # -------------------------
        coords = np.column_stack(np.where(binary))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        img_cropped = img.crop((x_min, y_min, x_max, y_max))

        # -------------------------
        # ✅ STEP 4: Add padding to center the digit
        # -------------------------
        img_padded = ImageOps.expand(img_cropped, border=20, fill=255)

        # -------------------------
        # ✅ STEP 5: Resize to MNIST size
        # -------------------------
        img_resized = img_padded.resize((28, 28))

        st.image(img_resized, caption="Preprocessed Image", width=200)

        # -------------------------
        # ✅ STEP 6: Convert to tensor
        # -------------------------
        img_tensor = transform(img_resized).unsqueeze(0)

        # -------------------------
        # ✅ STEP 7: Predict
        # -------------------------
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)

        st.success(f"✅ Predicted Digit: {predicted.item()}")

    st.success(f"✅ Predicted Digit: {predicted.item()}")
