import streamlit as st
import pandas as pd

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
        st.write('Projekt polegał na zbudowaniu sieci neuronowej, która potrafi wykrywać ludzkie twarze na zdjęciach, a także na filmach/gifach. Zbudowałyśmy także model rozpoznający (klasyfikujący) konkretne twarze, który wykorzystuje wiedzę na temat wykrywania dowolnych twarzy i jest rozszerzeniem zagadnienia detekcji twarzy. Na kolejnych zakładkach znajdują się kody źródłowe napisane w Pythonie w ramach projektu, a także możliwości przetestowania modeli.')

tabs = st.tabs(["Wykrywanie twarzy", "Klasyfikacja znanych twarzy", "Testuj na zdjęciu/wideo"])

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
        st.markdown('## 1.3 Załadowanie datasetu i ramek')
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
    transforms.ToPILImage(), # konwersja obrazu z numpy na obiekt PIL.Image
    transforms.Resize((256, 256)),  # zmiana rozmiaru na (256 x 256)
    transforms.ToTensor() # konwersja obrazu PIL na tensor pytorch (normalizowany do zakresu [0,1])
])

dataset = WiderFaceDataset(
    images_dir='data/WIDER_train/images', # ścieżka dokatalogu z obrazkami
    annotations=annotations, # adnotacje 
    transform=transform # przekształcenia
)
""")
        st.markdown('## 2.2 Tworzenie dataloadera')
        st.markdown("""
```python
# funkcja do grupowania danych w batchu: [(img1, target1), (img2, target2), ...] -> ([img1, img2, ...], [target1, target2, ...])
def collate_fn(batch):
    return tuple(zip(*batch))

# dataLoader, który będzie ładował dane do modelu w batchach (po 4 przykłady na raz)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
""")
        st.markdown('## 2.3 Wyświetlanie przykładowego batcha obrazów z bboxami')
        st.markdown("""
```python
batch = next(iter(dataloader)) # pobranie batcha (lista obrazów, lista targetów)
images, bboxes_batch = batch # rozpakowanie batcha na obrazy i odpowiadające im adnotacje 

fig, axs = plt.subplots(1, 4, figsize=(20,5)) # siatka 4 przykładowych obrazków
for i in range(4):
    img = images[i].permute(1, 2, 0).numpy()
    axs[i].imshow(img) # wyświetlenie obrazka
    axs[i].axis('off')
    
    bboxes = bboxes_batch[i]['boxes'].cpu().numpy() # pobranie bboxów (w formacie [xmin, ymin, xmax, ymax]) dla obrazka

    # dla każdego bboxa rysowany jest prostokąt na obrazie
    for box in bboxes:
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                 linewidth=2, edgecolor='r', facecolor='none')
        axs[i].add_patch(rect) # dodanie bboxa do wykresu
plt.show()
""")
        st.write('Przykładowy batch:')
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
print(f"\nUruchomiono na: {'GPU' if device.type == 'cuda' else 'CPU'}")
""")
        st.markdown('# 3. Podział danych i utworzenie dataloaderów')
        st.markdown("""
```python
# rozmiary zbiorów:
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
print("Rozmiar zbioru uczącego:", train_size)
print("Rozmiar zbioru walidacyjnego:", val_size)
print("Rozmiar zbioru testowego:", test_size)

# podział na zbiory danych — odpowiedzialne za ładowanie danych w batchach:
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# dataloadery:
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
""")
        st.write("Rozmiary zbiorów danych:")
        train_size = 6526
        val_size = 1398
        test_size = 1400
        co1, co2, co3 = st.columns([1,3,1])
        datasets_table = f"""
<table style="width:100%; border-collapse: collapse; font-size:16px;" border="1">
  <tr style="background-color:#2a5989; color:white; text-align: center;">
    <th style="text-align: center;">Zbiór uczący</th>
    <th style="text-align: center;">Zbiór walidacyjny</th>
    <th style="text-align: center;">Zbiór testowy</th>
  </tr>
  <tr style="background-color:#6ba6b7; color:black; text-align: center;">
    <td>{train_size}</td>
    <td>{val_size}</td>
    <td>{test_size}</td>
  </tr>
</table>
"""
        co2.markdown(datasets_table, unsafe_allow_html=True)
        st.markdown('# 4. Przygotowanie modelu i trening')
        st.markdown("""
```python
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT # wczytanie wag dla modelu Faster R-CNN z ResNet50 + FPN
model = fasterrcnn_resnet50_fpn(weights=weights) 

num_classes = 2  # ustawienie liczbęy klas — 1 klasa + tło 
in_features = model.roi_heads.box_predictor.cls_score.in_features # pobranie liczby wejściowych cech
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) # zamiana domyślnej warstwy na taką, która obsługuje 2 klasy

model.to(device) # przeniesienie modelu na GPU (jeśli dostępne), jeśli nie na CPU

# optymalizator i liczba epok 
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
num_epochs = 50

# listy do przechowywania strat z treningu i walidacji
train_losses = []
val_losses = []

# uczenie
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train", leave=False)
    for images, targets in train_loop:
        images = [img.to(device) for img in images] # przeniesienie danych na odpowiednie urządzenie
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets] 

        # pomijanie batchy bez boxów
        if not all(len(t["boxes"]) > 0 for t in targets):
            continue  

        optimizer.zero_grad()
        loss_dict = model(images, targets) # obliczanie strat
        losses = sum(loss for loss in loss_dict.values())
        losses.backward() # propagacja wsteczna
        optimizer.step() # aktualizaccja wag

        total_loss += losses.item()
        train_loop.set_postfix(loss=losses.item()) # pokazanie straty w pasku postępu

    avg_train_loss = total_loss / len(train_loader) # średnia strata z epoki

    # walidacja
    model.eval()  
    val_loss = 0

    with torch.no_grad():
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val", leave=False)
        for images, targets in val_loop:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # pomijanie batchy bez boxów
            if not all(len(t["boxes"]) > 0 for t in targets):
                val_loop.set_postfix(skip="empty targets")
                continue  

            loss_dict = model(images, targets) # obliczanie strat

            if isinstance(loss_dict, dict):
                losses = sum(loss for loss in loss_dict.values())
            else:
                val_loop.set_postfix(warning="model returned predictions, not loss")
                continue

            val_loss += losses.item()
            val_loop.set_postfix(val_loss=losses.item())

    # średnia strata
    if len(val_loader) > 0:
        avg_val_loss = val_loss / len(val_loader)
    else:
        avg_val_loss = 0.0

    # zapis strat
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# wykres strat
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Zbiór uczący')
plt.plot(val_losses, label='Zbiór walidacyjny')
plt.title('Wykres funkcji strat')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()
plt.show()
""")
        c1, c2, c3 = st.columns([1,3,1])
        c2.image('loss_plot.svg')
        st.markdown('# 5. Testowanie i wizualizacja wyników')
        st.markdown("""
```python
def compute_iou(box1, box2):

    # obliczanie współrzędnych prostokąta
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # obszar części wspólnej (intersection)
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # obszary pojedynczych bboxów
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # obszar łączny (union)
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area # IoU (Intersection over Union)

def evaluate(model, dataloader, device, iou_threshold=0.5, score_threshold=0.5):
    model.eval()
    precisions = []
    recalls = []
    f1_scores = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes'].cpu()
                pred_scores = output['scores'].cpu()
                gt_boxes = target['boxes'].cpu()
                matched_gt = set()

                tp = 0
                fp = 0

                for pred_box, score in zip(pred_boxes, pred_scores):
                    if score < score_threshold:
                        continue

                    best_iou = 0
                    best_gt_idx = -1

                    for j, gt_box in enumerate(gt_boxes):
                        if j in matched_gt:
                            continue
                        iou = compute_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = j

                    if best_iou >= iou_threshold:
                        tp += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        fp += 1

                fn = len(gt_boxes) - len(matched_gt)

                precision = tp / (tp + fp + 1e-6) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn + 1e-6) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall + 1e-6) if (precision + recall) > 0 else 0.0

                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f1 = sum(f1_scores) / len(f1_scores)

    print(f"Precyzja: {avg_precision:.4f}")
    print(f"Czułość: {avg_recall:.4f}")
    print(f"F1 score: {avg_f1:.4f}")

    return avg_precision, avg_recall, avg_f1

def visualize_predictions_vs_real(model, dataloader, device, score_threshold=0.5, max_images=20):
    model.eval()
    rows, cols = 4, 5
    count = 0
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i in range(len(images)):
                if count >= max_images:
                    for j in range(count, rows * cols):
                        axes[j].axis('off')
                    plt.tight_layout()
                    plt.show()
                    return

                img = images[i].permute(1, 2, 0).cpu().numpy()
                pred_boxes = outputs[i]['boxes'].detach().cpu().numpy()
                pred_scores = outputs[i]['scores'].detach().cpu().numpy()
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                ax = axes[count]
                ax.imshow(img)

                for box in gt_boxes:
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,linewidth=2, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)

                for box, score in zip(pred_boxes, pred_scores):
                    if score >= score_threshold:
                        x1, y1, x2, y2 = box
                        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,linewidth=2, edgecolor='lime', facecolor='none')
                        ax.add_patch(rect)
                        ax.text(x1, y1 - 5, f'{score:.2f}', color='lime', fontsize=6)

                ax.set_title(f"Zielone: przewidywane | Czerwone: rzeczywiste")
                ax.axis('off')
                count += 1

    for j in range(count, rows * cols):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def save_all_predictions(model, dataloader, device, score_threshold=0.5, output_dir='predykcje'):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i in range(len(images)):
                img = images[i].permute(1, 2, 0).cpu().numpy()
                pred_boxes = outputs[i]['boxes'].detach().cpu().numpy()
                pred_scores = outputs[i]['scores'].detach().cpu().numpy()
                gt_boxes = targets[i]['boxes'].cpu().numpy()

                fig, ax = plt.subplots(1, figsize=(6, 6))
                ax.imshow(img)

                for box in gt_boxes:
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                             linewidth=2, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)

                for box, score in zip(pred_boxes, pred_scores):
                    if score >= score_threshold:
                        x1, y1, x2, y2 = box
                        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                                 linewidth=2, edgecolor='lime', facecolor='none')
                        ax.add_patch(rect)
                        ax.text(x1, y1 - 5, f'{score:.2f}', color='lime', fontsize=6)

                ax.axis('off')
                plt.tight_layout()
                filepath = os.path.join(output_dir, f'predykcja_{count+1:04}.png')
                plt.savefig(filepath, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)
                count += 1

    print(f"Zapisano {count} obrazów do folderu '{output_dir}'.")

visualize_predictions_vs_real(model, test_loader, device)
evaluate(model, test_loader, device)
save_all_predictions(model, test_loader, device)
""")

        st.write('Przykładowe wyniki (czerwone - rzeczywiste, zielone - przewidywane):')
        z1, z2, z3, z4, z5, z6, z7 = st.columns([1,1,1,1,1,1,1])
        z1.image('1.png')
        z2.image('2.png')
        z3.image('3.png')
        z4.image('4.png')
        z5.image('5.png')
        z6.image('6.png')
        z7.image('7.png')
        z1.image('8.png')
        z2.image('9.png')
        z3.image('10.png')
        z4.image('11.png')
        z5.image('12.png')
        z6.image('13.png')
        z7.image('14.png')

        st.write("Wartości miar jakości dla modelu wykrywania twarzy:")
        precyzja = 0.7062 * 100
        czulosc = 0.8003 * 100
        f1 = 0.7243 * 100
        metrics_table = f"""
<table style="width:100%; border-collapse: collapse; font-size:16px;" border="1">
  <tr style="background-color:#2a5989; color:white; text-align: center;">
    <th style="text-align: center;">Precyzja</th>
    <th style="text-align: center;">Czułość</th>
    <th style="text-align: center;">F1 score</th>
  </tr>
  <tr style="background-color:#6ba6b7; color:black; text-align: center;">
    <td>{precyzja:.2f}%</td>
<td>{czulosc:.2f}%</td>
<td>{f1:.2f}%</td>
  </tr>
</table>
"""   
        col1, col2, col3 = st.columns([1,3,1])
        col2.markdown(metrics_table, unsafe_allow_html=True)
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
    st.markdown("<h1 style='text-align: center;'>Testuj na zdjęciu/wideo</h1>", unsafe_allow_html=True)
    
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

    model_file_id = "1535WMpVQNTlCGyPtE23kOg1duZXFLhen"
    model_path = "model-facedetect.pth"
    download_from_gdrive(model_file_id, model_path)

    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2) 

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    def detect_faces(image_np):
        original_h, original_w = image_np.shape[:2]
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (256, 256))
    
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
    ])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = transform(image_resized).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model([image_tensor])
    
        boxes = outputs[0]['boxes'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()
    
        scale_x = original_w / 256
        scale_y = original_h / 256
    
        boxes_scaled = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            boxes_scaled.append((x1, y1, x2, y2))
    
        filtered_boxes = [box for box, score in zip(boxes_scaled, scores) if score > 0.5]
        filtered_scores = [score for score in scores if score > 0.5]
    
        return filtered_boxes, filtered_scores

    f1, f2, f3 = st.columns([1,2,1])
    option = f2.selectbox("Wgraj:", ["zdjęcie", "wideo", "GIF"])

    if option == "zdjęcie":
        m1, m2, m3 = st.columns([1,3,1])
        m2.write("Wgraj zdjęcie lub wybierz przykład:")
        st.markdown(
        """
        <style>
        .stFileUploader label {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
        uploaded_file = m2.file_uploader("", type=["jpg", "jpeg", "png", "svg"])
        example_images = {
        "przykład 1": "img1.jpg",
        "przykład 2": "img2.jpg",
        "przykład 3": "img3.jpg",
        "przykład 4": "img4.jpg",
    }

        selected_example = None

        n1, n2, n3 = st.columns([1,5,1])
        with n2:
            cols = st.columns(len(example_images))  

            for col, (label, path) in zip(cols, example_images.items()):
                with col:
                    if st.button(f"{label}"):
                        selected_example = path

                    img = Image.open(path).convert("RGB")
                    st.image(img, use_container_width=True)

        image_np = None

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)

        elif selected_example:
            image = Image.open(selected_example).convert("RGB")
            image_np = np.array(image)

        if image_np is not None:
            boxes, scores = detect_faces(image_np)

            for (x1, y1, x2, y2), score in zip(boxes, scores):
                if score > 0.5:
                    cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

            col = st.columns([1,2,1])
            col[1].image(image_np, use_container_width=True)

    elif option == "wideo":
        m1, m2 = st.columns([1, 2])
        m1.write("Wgraj plik wideo:")
        st.markdown(
        """
        <style>
        .stFileUploader label {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
        video_file = m1.file_uploader("", type=["mp4", "avi", "mov"])

        example_videos = {
        "Wideo 1": "example1.mp4",
        "Wideo 2": "example2.mp4",
    }

        selected_video_example = None

        with m2:
            st.write("Wybierz przykład:")
            cols = st.columns(len(example_videos))

            for col, (label, path) in zip(cols, example_videos.items()):
                with col:
                    if st.button(label):
                        selected_video_example = path
                    st.video(path)

        video_path = None
        if video_file is not None:
            tfile = "temp_video.mp4"
            with open(tfile, 'wb') as f:
                f.write(video_file.read())
            video_path = tfile

        elif selected_video_example:
            video_path = selected_video_example

        if video_path:
            cap = cv2.VideoCapture(video_path)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                boxes, scores = detect_faces(frame)
                for (x1, y1, x2, y2), score in zip(boxes, scores):
                    if score > 0.5:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB", use_column_width=True)

            cap.release()

    elif option == "GIF":
        m1, m2, m3 = st.columns([1, 3, 1])
        m2.write("Wgraj plik GIF lub wybierz przykład:")
        st.markdown(
        """
        <style>
        .stFileUploader label {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
        gif_file = m2.file_uploader("", type=["gif"])

        example_gifs = {
        "GIF 1": "example1.gif",
        "GIF 2": "example2.gif",
    }

        selected_gif_example = None

        n1, n2, n3 = st.columns([1,5,1])
        with n2:
            st.write("Wybierz przykład:")
            cols = st.columns(len(example_gifs))

            for col, (label, path) in zip(cols, example_gifs.items()):
                with col:
                    if st.button(label):
                        selected_gif_example = path
                    st.image(path, use_column_width=True)

        gif_path = None
        if gif_file is not None:
            gif = Image.open(gif_file)
        elif selected_gif_example:
            gif = Image.open(selected_gif_example)
        else:
            gif = None

        if gif:
            stframe = st.empty()

            try:
                while True:
                    gif_frame = gif.convert("RGB")
                    frame_np = np.array(gif_frame)[:, :, ::-1].copy()

                    boxes, scores = detect_faces(frame_np)
                    for (x1, y1, x2, y2), score in zip(boxes, scores):
                        if score > 0.5:
                            cv2.rectangle(frame_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, channels="RGB", use_column_width=True)
                    gif.seek(gif.tell() + 1)

            except EOFError:
                pass







