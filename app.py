import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FastRCNNPredictor

# Załaduj model
@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.load_state_dict(torch.load("model-facedetect.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Transformacja obrazu
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def detect_faces(image_np):
    image_tensor = transform(image_np).unsqueeze(0)
    with torch.no_grad():
        prediction = model(image_tensor)[0]
    return prediction

# Interfejs Streamlit
st.title("🧠 Face Detection App")
option = st.radio("Wybierz tryb:", ["📷 Użyj kamery", "🖼️ Wgraj zdjęcie"])

if option == "🖼️ Wgraj zdjęcie":
    uploaded_file = st.file_uploader("Wgraj obraz", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        pred = detect_faces(image_np)

        for box, score in zip(pred['boxes'], pred['scores']):
            if score > 0.5:
                x1, y1, x2, y2 = box.int().numpy()
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)

        st.image(image_np, caption="Wykryte twarze", use_column_width=True)

elif option == "📷 Użyj kamery":
    st.warning("Streamlit nie wspiera jeszcze bezpośrednio kamery w przeglądarce w czasie rzeczywistym. Możesz użyć `streamlit-webrtc`.")

    st.code("➡️ Alternatywa: `streamlit-webrtc` pozwala na obsługę kamery. Chcesz wersję z tym modułem?")

