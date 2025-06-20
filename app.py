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
Zbiór danych pochodzi ze strony: [https://www.kaggle.com/datasets/iamprateek/wider-face-a-face-detection-dataset](https://www.kaggle.com/datasets/iamprateek/wider-face-a-face-detection-dataset).

Aby móc go wykorzystać w kodzie, należy mieć zapisany folder z plikami o następującej strukturze:
""")
        st.markdown("""
```python
data/
└── wider_face_split/
    └── wider_face_train_bbx_gt.txt
└── WIDER_train/
    └── images/
        └── 0--Parade
        └── 1--Handshaking
        └── ...
```
""")
        st.markdown("""
Do uczenia modelu w tym pliku wykorzystałyśmy 40 wybranych folderów.
""")
        st.markdown("""
```python
def parse_widerface_annotations(ann_file):
    annotations = {} # słownik do przechowywania adnotacji postaci: {ścieżka do obrazu: [lista bboxów]}
    with open(ann_file, 'r') as f:
        lines = f.readlines() # wczytuje wszystkie linie w pliku adnotacji
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # jeśli linia kończy się na '.jpg', oznacza to początek nowego wpisu z adnotacjami dla danego obrazu
        if line.endswith('.jpg'):
            img_path = line
            i += 1
            num_faces = int(lines[i].strip()) # liczba twarzy na obrazku
            i += 1
            bboxes = [] # lista do przechowywania ramek (bbox) dla danego obrazu

            # iteracja po liczbie twarzy i pobranie bboxów
            for _ in range(num_faces):
                parts = lines[i].strip().split() # podział linii na części
                x, y, w, h = map(int, parts[:4]) # pobranie współrzędnych i wymiarów bboxa
                bboxes.append([x, y, w, h]) # dodanie do listy
                i += 1
            annotations[img_path] = bboxes # dodanie adnotacji dla danego obrazu do słownika
        else:
            i += 1 # przejście do kolejnej linii, jeśli nie znaleziono obrazka
    return annotations

ANNOTATIONS_PATH = 'data/wider_face_split/wider_face_train_bbx_gt.txt' # ścieżka do pliku z adnotacjami
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
        self.transform = transform

        # lista istniejących plików i odpowiadające im poprawne adnotacje
        self.image_files = []
        self.valid_annotations = {}

        # filtrowanie plików, zostawianie tylko tych, które istnieją na dysku
        for img_filename, bboxes in annotations.items():
            img_path = os.path.join(images_dir, img_filename)
            if os.path.exists(img_path):
                self.image_files.append(img_filename)
                self.valid_annotations[img_filename] = bboxes
            else:
                None # jeśli plik nie istnieje, ignorujemy go
                
    def __len__(self):
        return len(self.image_files) # zwraca liczbę dostępnych obrazków

    def __getitem__(self, idx):

        # pobranie nazwy pliku i ścieżki
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_filename)

        # wczytanie obrazu i konwersja na RGB
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2] # oryginalne rozmiary danego obrazka
        resized_h, resized_w = 256, 256 # wymair docelowy
        image_resized = cv2.resize(image, (resized_w, resized_h))

        # współczynniki skalowania bboxów
        scale_x = resized_w / orig_w
        scale_y = resized_h / orig_h

        bboxes = self.valid_annotations[img_filename]  # pobranie oryginalnych bboxów (lista [x,y,w,h])
        boxes = []

        # skalowanie bboxów i przekształcenie do formatu [xmin, ymin, xmax, ymax]
        for x, y, w, h in bboxes:
            if w < 2 or h < 2:
                continue  # pomija zbyt małe lub zerowe bboxy

            xmin = x * scale_x
            ymin = y * scale_y
            xmax = (x + w) * scale_x
            ymax = (y + h) * scale_y

            if xmax <= xmin or ymax <= ymin:
                continue  # dodatkowe zabezpieczenie przed błędnymi wartościami

            boxes.append([xmin, ymin, xmax, ymax])

        # jeśli brak poprawnych boxów, losowany jest inny przykład
        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self))

        boxes = torch.tensor(boxes, dtype=torch.float32) # zamiana listy ramek na tensor
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # przypisanie twarzom etykiety 1 

        # przekształcenie obrazu do tensora 
        if self.transform:
            image_tensor = self.transform(image_resized)
        else:
            transform_default = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])
            image_tensor = transform_default(image_resized)

        # zwracanie danych w formacie zgodnym z detekcją obiektów w PyTorch
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        return image_tensor, target # zwraca przetworzony obraz i odpowiadające mu dane (ramki i etykiety)
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
plt.show())
""")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image('svg1.svg')

        with col2:
            st.image('svg2.svg')

        with col3:
            st.image('svg3.svg')

        with col4:
            st.image('svg4.svg')

        st.markdown('## 2.4 Sprawdzenie dostępności GPU')
        st.markdown("""
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n Uruchomiono na: {'GPU' if device.type == 'cuda' else 'CPU'}")
""")
        st.markdown('# 3. Podział danych i utworzenie dataloaderów')
        st.markdown("""
```python
# rozmiary podzbiorów:
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

# podział:
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# dataloadery:
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
""")
        st.markdown('# 4. Przygotowanie modelu i trening')
        st.markdown("""
```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)

num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
num_epochs = 50

train_losses = []
val_losses = []

# trening:
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train", leave=False)
    for images, targets in train_loop:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        train_loop.set_postfix(loss=losses.item())

    avg_train_loss = total_loss / len(train_loader)

    # walidacja:
    model.eval() 
    val_loss = 0

    with torch.no_grad():
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val", leave=False)
        for images, targets in val_loop:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
            val_loop.set_postfix(val_loss=losses.item())

    avg_val_loss = val_loss / len(val_loader)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
""")
        st.markdown('# 5. Testowanie i wizualizacja wyników')
        st.markdown("""
```python

""")
        st.markdown('# 6. Zapisanie modelu')
        st.markdown("""
```python
torch.save(model.state_dict(), "model-facedetect.pth")
torch.save(model, "model-facedetect-full.pth")
""")
        

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

    def download_from_gdrive(file_id, output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        if not os.path.exists(output_path):
            gdown.download(url, output_path, quiet=False)

    model_file_id = "1_XyIO6P37cPoneyClBHFz9ZYzB1g15Z7"
    model_path = "model-facedetect.pth"
    download_from_gdrive(model_file_id, model_path)

    model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model.eval()

    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()])

    def detect_faces(image_np):
        original_h, original_w = image_np.shape[:2]

        image_resized = cv2.resize(image_np, (256, 256))  # rozmiar zgodny z transform
        image_tensor = transform(image_resized).unsqueeze(0)

        with torch.no_grad():
            prediction = model(image_tensor)[0]

        scale_x = original_w / 256
        scale_y = original_h / 256

        boxes_scaled = []
        for box in prediction['boxes']:
            x1, y1, x2, y2 = box.tolist()
            x1 = int(x1 * scale_x)
            x2 = int(x2 * scale_x)
            y1 = int(y1 * scale_y)
            y2 = int(y2 * scale_y)
            boxes_scaled.append((x1, y1, x2, y2))

        return boxes_scaled, prediction['scores']

    uploaded_file = st.file_uploader("Wgraj obraz:", type=["jpg", "jpeg", "png", "svg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        boxes, scores = detect_faces(image_np)

        for (x1, y1, x2, y2), score in zip(boxes, scores):
            if score > 0.5:
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

        st.image(image_np, caption="Wykryte twarze", use_container_width=True)


with tabs[3]:
    st.markdown("<h1 style='text-align: center;'>Testuj na żywo</h1>", unsafe_allow_html=True)

    from streamlit_webrtc import VideoTransformerBase
    import cv2
    import torch
    from streamlit_webrtc import webrtc_streamer

    class FaceDetector(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
            original_h, original_w = img.shape[:2]
            img_resized = cv2.resize(img_rgb, (256, 256))
            image_tensor = transform(img_resized).unsqueeze(0)
        
            with torch.no_grad():
                prediction = model(image_tensor)[0]
        
            scale_x = original_w / 256
            scale_y = original_h / 256
        
            for box, score in zip(prediction['boxes'], prediction['scores']):
                if score > 0.5:
                    x1, y1, x2, y2 = box.tolist()
                    x1 = int(x1 * scale_x)
                    x2 = int(x2 * scale_x)
                    y1 = int(y1 * scale_y)
                    y2 = int(y2 * scale_y)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
            return img

    webrtc_streamer(key="face-detection", video_processor_factory=FaceDetector)




