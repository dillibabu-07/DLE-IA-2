from pathlib import Path
import numpy as np
import torch
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance
from src.dbn_pattern_recognition import DBNClassifier, MODEL_DIR

class ToTensor:
    def __call__(self, image):
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array.reshape(1, 28, 28)).float()

transforms = type("Transforms", (), {"ToTensor": staticmethod(lambda: ToTensor())})()

st.set_page_config(page_title="DBN Pattern Recognition", page_icon="🔢", layout="centered")
st.title("Deep Belief Network – Pattern Recognition")
st.write("Upload a handwritten digit image (0–9). The application uses a DBN pretrained with stacked Restricted Boltzmann Machines.")
model_path = MODEL_DIR / "dbn_mnist.pt"
if not model_path.exists():
    st.warning("Model not found. Train it first with: `python src/dbn_pattern_recognition.py`")
    st.stop()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model = DBNClassifier(tuple(checkpoint["layer_sizes"]), checkpoint.get("n_classes", 10)).to(device)
model.load_state_dict(checkpoint["state_dict"]); model.eval()
upload = st.file_uploader("Choose a digit image", type=["png", "jpg", "jpeg", "bmp"])
if upload:
    image = Image.open(upload).convert("L")
    st.image(image, caption="Uploaded image", width=220)
    # MNIST uses light digits on a dark background; invert common dark-ink-on-white uploads.
    small = ImageOps.fit(image, (28, 28))
    if sum(small.resize((1, 1)).getpixel((0, 0)) for _ in [0]) > 127:
        small = ImageOps.invert(small)
    tensor = transforms.ToTensor()(small).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor.to(device)), dim=1)[0].cpu()
    predicted = int(probabilities.argmax())
    st.success(f"Predicted digit: **{predicted}**")
    st.write(f"Model confidence: **{probabilities[predicted].item()*100:.2f}%**")
    st.bar_chart({str(i): float(probabilities[i]) for i in range(10)})
    st.caption("Predictions may be less reliable for photographs or handwriting unlike MNIST. Use a centered, high-contrast digit for best results.")
