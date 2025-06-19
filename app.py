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
            rgba(193, 18, 31, 0.7),    /* głęboki szkarłat #C1121F */
            rgba(160, 30, 40, 0.6),    /* ciepła ciemniejsza czerwień */
            rgba(90, 15, 25, 0.45)     /* przygaszony ciemnoczerwony */
        );
        border: 1px solid rgba(193, 18, 31, 0.5);
        border-radius: 20px;
        padding: 24px;
        box-shadow:
            0 0 16px rgba(193, 18, 31, 0.45),
            0 4px 26px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(14px) brightness(0.95);
        background-blend-mode: overlay;
        transition: all 0.3s ease-in-out;
    }

    div[data-testid="stVerticalBlock"]:has(div#gradient_container_marker):not(:has(div#outer_marker)):hover {
        transform: translateY(-6px);
        box-shadow:
            0 0 22px rgba(193, 18, 31, 0.65),
            0 8px 34px rgba(0, 0, 0, 0.6);
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
        st.write('Projekt polegał na zbudowaniu sieci neuronowej, która potrafi wykrywać ludzkie twarze na zdjęciach, a także w czasie rzeczywistym, np. z kamery w laptopie. Zbudowałyśmy także model rozpoznający (klasyfikujący) konkretne twarze, który wykorzystuje wiedzę na temat wykrywania dowolnych twarzy i jest rozszerzeniem zagadnienia detekcji twarzy. Na kolejnych zakładkach znajdują się kody źródłowe napisane w Pythonie w ramach projektu, a także możliwości przetestowania modeli.')

tabs = st.tabs(["Wykrywanie twarzy", "Klasyfikacja znanych twarzy", "Testuj na zdjęciu", "Testuj na żywo"])

with tabs[0]:
    st.markdown("<h1 style='text-align: center;'>Wykrywanie twarzy - kod</h1>", unsafe_allow_html=True)
    st.markdown('')

    styled_container = st.container()
    st.markdown("<div id='outer_marker'></div>", unsafe_allow_html=True)
    with styled_container:
        st.markdown("<div id='gradient_container_marker'></div>", unsafe_allow_html=True)

        st.markdown('# 1. Przygotowanie środowiska i danych')
        st.markdown('## 1.1 Instalacja i import bibliotek')
        st.markdown("""
```python
!pip install torch torchvision opencv-python matplotlib
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import numpy as np
from torchvision import transforms
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.optim as optim
from tqdm import tqdm
""")
        st.markdown('## 1.2 Parsowanie pliku adnotacji WIDER Face')
        st.markdown("""
```python
def parse_widerface_annotations(ann_file):
    annotations = {}
    with open(ann_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.endswith('.jpg'):
            img_path = line
            i += 1
            num_faces = int(lines[i].strip())
            i += 1
            bboxes = []
            for _ in range(num_faces):
                parts = lines[i].strip().split()
                x, y, w, h = map(int, parts[:4])
                bboxes.append([x, y, w, h])
                i += 1
            annotations[img_path] = bboxes
        else:
            i += 1
    return annotations

ANNOTATIONS_PATH = 'data/wider_face_split/wider_face_train_bbx_gt.txt'
annotations = parse_widerface_annotations(ANNOTATIONS_PATH)
print(f'Liczba obrazów: {len(annotations)}')
print(f'Liczba obrazów z adnotacjami: {len(annotations)}')
print('Przykład bboxów dla pierwszego obrazu:', list(annotations.values())[0])
""")
        st.markdown('## 1.3 Załdaowanie datasetu i ramek')
        st.markdown("""
```python
class WiderFaceDataset(Dataset):
    def __init__(self, images_dir, annotations, transform=None):
        self.images_dir = images_dir
        self.annotations = annotations
        self.transform = transform
        self.image_files = list(annotations.keys())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_filename)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f'Nie znaleziono obrazu: {img_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]
        resized_h, resized_w = 256, 256
        image_resized = cv2.resize(image, (resized_w, resized_h))

        scale_x = resized_w / orig_w
        scale_y = resized_h / orig_h

        bboxes = self.annotations[img_filename]  # lista [x,y,w,h]
        boxes = []
        for x, y, w, h in bboxes:
            if w < 2 or h < 2:
                continue  # pomiń zbyt małe lub zerowe bboxy

            xmin = x * scale_x
            ymin = y * scale_y
            xmax = (x + w) * scale_x
            ymax = (y + h) * scale_y

            if xmax <= xmin or ymax <= ymin:
                continue  

            boxes.append([xmin, ymin, xmax, ymax])

        # jeśli żaden box nie był poprawny, próbuj inny przykład
        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        if self.transform:
            image_tensor = self.transform(image_resized)
        else:
            transform_default = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])
            image_tensor = transform_default(image_resized)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        return image_tensor, target
""")
        st.markdown('# 2. Przetwarzanie danych i przygotowanie do uczenia')
        st.markdown('## 2.1 Definicja transformacji i inicjalizacja datasetu')
        st.markdown("""
```python
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),  # ustalony rozmiar
    transforms.ToTensor()
])

dataset = WiderFaceDataset(
    images_dir='data/WIDER_train/images',
    annotations=annotations,
    transform=transform
)
""")
        st.markdown('## 2.2 Tworzenie dataloadera')
        st.markdown("""
```python
def collate_fn(batch):
    return tuple(zip(*batch))

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
)
""")
        st.markdown('## 2.3 Wyświetlanie przykładowego batcha obrazów z bboxami')
        st.markdown("""
```python
batch = next(iter(dataloader))
images, bboxes_batch = batch 

fig, axs = plt.subplots(1, 4, figsize=(20,5))
for i in range(4):
    img = images[i].permute(1, 2, 0).numpy()
    axs[i].imshow(img)
    axs[i].axis('off')
    
    bboxes = bboxes_batch[i]['boxes'].cpu().numpy()
    for box in bboxes:
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                 linewidth=2, edgecolor='r', facecolor='none')
        axs[i].add_patch(rect)
plt.show()
)
""")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(svg1, unsafe_allow_html=True)

        with col2:
            st.markdown(svg2, unsafe_allow_html=True)

        with col3:
            st.markdown(svg3, unsafe_allow_html=True)

        with col4:
            st.markdown(svg4, unsafe_allow_html=True)
        

with tabs[1]:
    st.markdown("<h1 style='text-align: center;'>Klasyfikacja znanych twarzy - kod</h1>", unsafe_allow_html=True)
    st.markdown('')

    styled_container = st.container()
    st.markdown("<div id='outer_marker'></div>", unsafe_allow_html=True)
    with styled_container:
        st.markdown("<div id='gradient_container_marker'></div>", unsafe_allow_html=True)

        st.markdown("""
```python
""")

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

    #def download_from_gdrive(file_id, output_path):
        #url = f"https://drive.google.com/uc?id={file_id}"
        #if not os.path.exists(output_path):
            #gdown.download(url, output_path, quiet=False)

    #model_file_id = "1HYWwhDrrUvL66EtmWRn3kycHYdWN1Bzz"
    #model_path = "model-facedetect.pth"

    #download_from_gdrive(model_file_id, model_path)

    #model = fasterrcnn_resnet50_fpn(pretrained=False)
    #num_classes = 2
    #in_features = model.roi_heads.box_predictor.cls_score.in_features
    #model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    #state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    #model.load_state_dict(state_dict)
    #model.eval()

    #transform = transforms.Compose([
    #transforms.ToPILImage(),
    #transforms.Resize((256, 256)),
    #transforms.ToTensor()
    #])

    #def detect_faces(image_np):
     #   image_tensor = transform(image_np).unsqueeze(0)
      #  with torch.no_grad():
       #     prediction = model(image_tensor)[0]
        #return prediction

    #uploaded_file = st.file_uploader("Wgraj obraz:", type=["jpg", "jpeg", "png"])
    #if uploaded_file is not None:
     #   image = Image.open(uploaded_file).convert("RGB")
      #  image_np = np.array(image)
       # pred = detect_faces(image_np)

        #for box, score in zip(pred['boxes'], pred['scores']):
         #   if score > 0.5:
          #      x1, y1, x2, y2 = box.int().numpy()
           #     cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #st.image(image_np, caption="Wykryte twarze", use_column_width=True)

with tabs[3]:
    st.markdown("<h1 style='text-align: center;'>Testuj na żywo</h1>", unsafe_allow_html=True)

    #class FaceDetector(VideoTransformerBase):
     #   def transform(self, frame):
      #      img = frame.to_ndarray(format="bgr24")
       #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #    pred = detect_faces(img_rgb)

         #   for box, score in zip(pred['boxes'], pred['scores']):
          #      if score > 0.5:
           #         x1, y1, x2, y2 = box.int().numpy()
            #        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #return img

    #webrtc_streamer(key="face-detection", video_processor_factory=FaceDetector)


