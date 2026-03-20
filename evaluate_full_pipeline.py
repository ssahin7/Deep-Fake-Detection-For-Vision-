import os
import cv2
import torch
import mediapipe as mp
import numpy as np
from torchvision import transforms, models
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = r"D:\proje klasörleri\CV_DeepFake\HDF_DataSet_Balanced\test"
SAVE_DIR = r"outputs_ensemble"

# ----------------------------
# TRANSFORM
# ----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ----------------------------
# FACEMESH GEOMETRY
# ----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def extract_geometry(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return np.zeros(4)
    landmarks = results.multi_face_landmarks[0].landmark
    def dist(p1,p2):
        return np.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)
    eye_dist = dist(landmarks[33],landmarks[263])
    mouth_width = dist(landmarks[61],landmarks[291])
    nose_chin = dist(landmarks[1],landmarks[152])
    face_width = dist(landmarks[234],landmarks[454])
    eye_width = dist(landmarks[33],landmarks[133])
    return np.array([eye_dist/face_width, mouth_width/face_width, nose_chin/face_width, eye_width/face_width])

# ----------------------------
# FFT
# ----------------------------
def compute_fft(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20*np.log(np.abs(fshift)+1e-8)
    magnitude -= magnitude.min()
    magnitude /= (magnitude.max()+1e-8)
    magnitude = (magnitude*255).astype(np.uint8)
    magnitude = cv2.resize(magnitude,(224,224))
    magnitude = np.stack([magnitude]*3,axis=-1)
    return magnitude

# ----------------------------
# DATASET
# ----------------------------
def load_dataset(path):
    samples = []
    for label_name, label_val in [("real",0),("fake",1)]:
        class_path = os.path.join(path,label_name)
        for f in os.listdir(class_path):
            if f.lower().endswith((".jpg",".png",".jpeg")):
                samples.append({"image_path":os.path.join(class_path,f),"label":label_val})
    return samples

dataset = load_dataset(DATA_ROOT)

# ----------------------------
# MODELS
# ----------------------------
from torchvision.models import vit_b_16, ViT_B_16_Weights, efficientnet_b0, EfficientNet_B0_Weights

class RGBModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad=False
        self.model.heads.head = nn.Sequential(
            nn.Linear(self.model.heads.head.in_features,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,1)
        )
    def forward(self,x):
        return self.model(x)

class FFTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad=False
        self.model.classifier[1] = nn.Sequential(
            nn.Linear(self.model.classifier[1].in_features,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,1)
        )
    def forward(self,x):
        return self.model(x)

class GeoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4,32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
    def forward(self,x):
        return self.model(x)

def load_model(model_class, path):
    model = model_class().to(DEVICE)
    model.load_state_dict(torch.load(path,map_location=DEVICE))
    model.eval()
    return model

rgb_model = load_model(RGBModel, os.path.join(SAVE_DIR,"best_rgb.pth"))
fft_model = load_model(FFTModel, os.path.join(SAVE_DIR,"best_fft.pth"))
geo_model = load_model(GeoModel, os.path.join(SAVE_DIR,"best_geo.pth"))

# ----------------------------
# EVALUATE
# ----------------------------
y_true=[]
y_pred=[]
y_scores=[]  # ensemble score

with torch.no_grad():
    for item in dataset:
        img=cv2.imread(item["image_path"])
        rgb = transform(img).unsqueeze(0).to(DEVICE)
        fft = transform(compute_fft(img)).unsqueeze(0).to(DEVICE)
        geo = torch.tensor(extract_geometry(img),dtype=torch.float32).unsqueeze(0).to(DEVICE)

        rgb_out = torch.sigmoid(rgb_model(rgb))
        fft_out = torch.sigmoid(fft_model(fft))
        geo_out = torch.sigmoid(geo_model(geo))

        final = 0.5*rgb_out + 0.3*fft_out + 0.2*geo_out
        pred = (final>0.5).int().item()

        y_true.append(item["label"])
        y_pred.append(pred)
        y_scores.append(final.item())

# ----------------------------
# METRICS
# ----------------------------
precision = precision_score(y_true,y_pred)
recall = recall_score(y_true,y_pred)
f1 = f1_score(y_true,y_pred)
acc = accuracy_score(y_true,y_pred)

print("===== ENSEMBLE TEST RESULT =====")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ----------------------------
# CONFUSION MATRIX
# ----------------------------
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real","Fake"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# ----------------------------
# ROC CURVE
# ----------------------------
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# ----------------------------
# PRECISION-RECALL CURVE
# ----------------------------
precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
plt.figure()
plt.plot(recall_vals, precision_vals, lw=2, color='green')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()