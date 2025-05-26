import streamlit as st
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FastRCNNPredictor
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ======= MODEL LOADING =======
@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.load_state_dict(torch.load("model-facedetect.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# ======= TRANSFORM IMAGE =======
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


# ======= STREAMLIT UI =======
st.title("📷 Real-Time Face Detection App")

mode = st.radio("Wybierz tryb:", ["🖼️ Wgraj zdjęcie", "📹 Kamera (real-time)"])

# ======= IMAGE UPLOAD =======
if mode == "🖼️ Wgraj zdjęcie":
    uploaded_file = st.file_uploader("Wgraj obraz:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        pred = detect_faces(image_np)

        for box, score in zip(pred['boxes'], pred['scores']):
            if score > 0.5:
                x1, y1, x2, y2 = box.int().numpy()
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

        st.image(image_np, caption="Wykryte twarze", use_column_width=True)


# ======= WEBCAM STREAM =======
elif mode == "📹 Kamera (real-time)":

    class FaceDetector(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred = detect_faces(img_rgb)

            for box, score in zip(pred['boxes'], pred['scores']):
                if score > 0.5:
                    x1, y1, x2, y2 = box.int().numpy()
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            return img

    webrtc_streamer(key="face-detection", video_processor_factory=FaceDetector)

