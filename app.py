import streamlit as st

st.set_page_config(page_title="Wstęp do sieci neuronowych - projekt", layout="wide")

gradient_css = """
<style>
.stApp {
    background: linear-gradient(180deg, #2a5989 0%, #6ba6b7 100%);
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
"""

st.markdown(gradient_css, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .emoji-top {
        margin-top: 0px; 
    }
    </style>
    
    <div style="text-align: center;">
        <h1>Wykrywanie i rozpoznwanie twarzy</h1>
    </div>
    """, 
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlock"]:has(div#gradient_container_marker):not(:has(div#outer_marker)) {
        background: linear-gradient(
            135deg,
            rgba(180, 68, 251, 0.25),
            rgba(45, 3, 94, 0.2),
            rgba(32, 33, 37, 0.1)
        );
        border: 1px solid rgba(180, 68, 251, 0.4);
        border-radius: 20px;
        padding: 24px;
        box-shadow:
            0 0 10px rgba(180, 68, 251, 0.25),
            0 4px 20px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(14px) brightness(1.1);
        background-blend-mode: overlay;
        transition: all 0.3s ease-in-out;
    }

    div[data-testid="stVerticalBlock"]:has(div#gradient_container_marker):not(:has(div#outer_marker)):hover {
        transform: translateY(-6px);
        box-shadow:
            0 0 18px rgba(180, 68, 251, 0.4),
            0 8px 30px rgba(0, 0, 0, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True
)

cols = st.columns((1,2,1))

with cols[1]:
    styled_container = st.container()
    st.markdown("<div id='outer_marker'></div>", unsafe_allow_html=True)
    with styled_container:
        st.markdown("<div id='gradient_container_marker'></div>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>Opis projektu</h2>", unsafe_allow_html=True)
        st.markdown('')
        st.write('Celem naszego projektu było zbudowanie sieci neuronowej, która potrafi wykrywać ludzkie twarze na obrazach, a także w czasie rzeczywistym, np. z kamery w laptopie. Zbudowałyśmy także model rozpoznający (klasyfikujący) konkretne twarze, który wykorzystuje wiedzę na temat wykrywania dowolnych twarzy i jest rozszerzeniem zagadnienia detekcji twarzy.')

tabs = st.tabs(["Wykrywanie twarzy", "Klasyfikacja znanych twarzy", "Testuj na zdjęciu", "Testuj na żywo"])

with tabs[0]:
    st.markdown("<h1 style='text-align: center;'>Wykrywanie twarzy</h1>", unsafe_allow_html=True)
    st.markdown('')

with tabs[1]:
    st.markdown("<h1 style='text-align: center;'>Klasyfikacja znanych twarzy</h1>", unsafe_allow_html=True)
    st.markdown('')

with tabs[2]:
    st.markdown("<h1 style='text-align: center;'>Testuj na zdjęciu</h1>", unsafe_allow_html=True)
    st.markdown('')

    import os
    import gdown
    import torch
    from torchvision import transforms
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    import numpy as np
    import cv2
    from PIL import Image
    import streamlit as st

    def download_from_gdrive(file_id, output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        if not os.path.exists(output_path):
            gdown.download(url, output_path, quiet=False)

    model_file_id = "1HYWwhDrrUvL66EtmWRn3kycHYdWN1Bzz"
    model_path = "model-facedetect.pth"

    download_from_gdrive(model_file_id, model_path)

    model = fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

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

    uploaded_file = st.file_uploader("Wgraj obraz:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        pred = detect_faces(image_np)

        for box, score in zip(pred['boxes'], pred['scores']):
            if score > 0.5:
                x1, y1, x2, y2 = box.int().numpy()
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        st.image(image_np, caption="Wykryte twarze", use_column_width=True)

with tabs[3]:
    st.markdown("<h1 style='text-align: center;'>Testuj na żywo</h1>", unsafe_allow_html=True)

    import torch
    import numpy as np
    import cv2
    from PIL import Image
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    @st.cache_resource
    def load_model():
        model = fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        model.load_state_dict(torch.load("model-facedetect.pth", map_location=torch.device('cpu')))
        model.eval()
        return model

    model = load_model()

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


