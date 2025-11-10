import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
from streamlit_drawable_canvas import st_canvas

# -------------------------
# ✅ CNN MODEL
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
# ✅ Load Model
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

# ===========================================================
# ✅ IMAGE PREPROCESSOR
# ===========================================================
def preprocess_image(img):
    img = ImageOps.invert(img)
    img_np = np.array(img)
    binary = img_np < 200

    if not binary.any():
        return None

    coords = np.column_stack(np.where(binary))
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    img = img.crop((x_min, y_min, x_max, y_max))
    img = ImageOps.expand(img, border=20, fill=255)
    img = img.resize((28, 28))

    return img

# ===========================================================
# ✅ STREAMLIT UI
# ===========================================================
st.title("🧠 MNIST Digit Classifier (Upload + Canvas + Preprocessing)")

option = st.radio("Choose Input Method:", ["Upload Image", "Draw on Canvas"])

# ===========================================================
# ✅ UPLOAD IMAGE
# ===========================================================
if option == "Upload Image":
    uploaded = st.file_uploader("Upload digit image", type=["png", "jpg", "jpeg"])

    if uploaded:
        original = Image.open(uploaded).convert("L")
        st.image(original, caption="Original Image", width=200)

        processed = preprocess_image(original)

        if processed is None:
            st.error("Digit not detected. Try a clearer photo.")
        else:
            st.image(processed, caption="Preprocessed Image", width=200)
            img_tensor = transform(processed).unsqueeze(0)

            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output, 1)

            st.success(f"✅ Predicted Digit: {predicted.item()}")

# ===========================================================
# ✅ DRAW ON CANVAS
# ===========================================================
if option == "Draw on Canvas":
    st.write("Draw a digit below:")

    canvas = st_canvas(
        fill_color="#00000000",
        stroke_width=12,
        stroke_color="black",
        background_color="white",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas.image_data is not None:
        img = Image.fromarray(canvas.image_data.astype("uint8")).convert("L")
        processed = preprocess_image(img)

        if processed:
            st.image(processed, caption="Processed Canvas Image", width=200)
            img_tensor = transform(processed).unsqueeze(0)

            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output, 1)

            st.success(f"✅ Predicted Digit: {predicted.item()}")
        else:
            st.info("Draw a digit clearly for prediction.")
