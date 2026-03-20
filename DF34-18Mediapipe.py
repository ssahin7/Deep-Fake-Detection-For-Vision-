import os
import cv2
import torch
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


DATA_ROOT = r"D:\proje klasörleri\CV_DeepFake\HDF_DataSet_Balanced"
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "outputs_ensemble"
os.makedirs(SAVE_DIR, exist_ok=True)


# FACEMESH

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


# FFT

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


# TRANSFORM

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


# DATASET

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir):
        self.samples=[]
        for label_name in ["real","fake"]:
            label = 0 if label_name=="real" else 1
            folder=os.path.join(root_dir,label_name)
            for img in os.listdir(folder):
                self.samples.append((os.path.join(folder,img),label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        path,label=self.samples[idx]
        img=cv2.imread(path)
        rgb=transform(img)
        fft_img=compute_fft(img)
        freq=transform(fft_img)
        geo=torch.tensor(extract_geometry(img),dtype=torch.float32)
        return rgb,freq,geo,label


# MODELS

# RGB → ViT
from torchvision.models import vit_b_16, ViT_B_16_Weights
class RGBModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.heads.head = nn.Sequential(
            nn.Linear(self.model.heads.head.in_features,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,1)
        )
    def forward(self,x):
        return self.model(x)

# FFT → EfficientNet-B0
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
class FFTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier[1] = nn.Sequential(
            nn.Linear(self.model.classifier[1].in_features,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,1)
        )
    def forward(self,x):
        return self.model(x)

# Geo → MLP
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


# METRICS

def compute_metrics(y_true,y_pred):
    precision=precision_score(y_true,y_pred)
    recall=recall_score(y_true,y_pred)
    f1=f1_score(y_true,y_pred)
    acc=accuracy_score(y_true,y_pred)
    intersection=((y_true==1)&(y_pred==1)).sum()
    union=((y_true==1)|(y_pred==1)).sum()
    iou=intersection/(union+1e-6)
    return precision,recall,f1,acc,iou


# TRAIN 

def train_model(model,train_loader,val_loader,mode):
    model=model.to(DEVICE)
    optimizer=optim.AdamW(model.parameters(),lr=LR)
    criterion=nn.BCEWithLogitsLoss()
    best_val=float('inf')
    for epoch in range(1,EPOCHS+1):
        model.train()
        train_loss=0
        for rgb,freq,geo,labels in tqdm(train_loader,desc=f"[{mode.upper()} TRAIN] Epoch {epoch}"):
            rgb,freq,geo=rgb.to(DEVICE),freq.to(DEVICE),geo.to(DEVICE)
            labels=labels.float().unsqueeze(1).to(DEVICE)
            optimizer.zero_grad()
            if mode=="rgb":
                outputs=model(rgb)
            elif mode=="fft":
                outputs=model(freq)
            else:
                outputs=model(geo)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
        train_loss/=len(train_loader)

        model.eval()
        val_loss=0
        with torch.no_grad():
            for rgb,freq,geo,labels in val_loader:
                rgb,freq,geo=rgb.to(DEVICE),freq.to(DEVICE),geo.to(DEVICE)
                labels=labels.float().unsqueeze(1).to(DEVICE)
                if mode=="rgb":
                    outputs=model(rgb)
                elif mode=="fft":
                    outputs=model(freq)
                else:
                    outputs=model(geo)
                loss=criterion(outputs,labels)
                val_loss+=loss.item()
        val_loss/=len(val_loader)
        print(f"[{mode.upper()}] Epoch {epoch} | Train {train_loss:.4f} | Val {val_loss:.4f}")
        if val_loss<best_val:
            best_val=val_loss
            torch.save(model.state_dict(),os.path.join(SAVE_DIR,f"best_{mode}.pth"))
    return model


train_loader=DataLoader(DeepfakeDataset(os.path.join(DATA_ROOT,"train")),batch_size=BATCH_SIZE,shuffle=True)
val_loader=DataLoader(DeepfakeDataset(os.path.join(DATA_ROOT,"val")),batch_size=BATCH_SIZE)
test_loader=DataLoader(DeepfakeDataset(os.path.join(DATA_ROOT,"test")),batch_size=BATCH_SIZE)


# TRAIN

print("===== RGB TRAINING =====")
rgb_model=train_model(RGBModel(),train_loader,val_loader,"rgb")

print("===== FFT TRAINING =====")
fft_model=train_model(FFTModel(),train_loader,val_loader,"fft")

print("===== GEO TRAINING =====")
geo_model=train_model(GeoModel(),train_loader,val_loader,"geo")


# BOOSTED

def ensemble_test(rgb_model,fft_model,geo_model,loader):
    rgb_model.eval()
    fft_model.eval()
    geo_model.eval()
    preds_all=[]
    labels_all=[]
    with torch.no_grad():
        for rgb,freq,geo,labels in loader:
            rgb,freq,geo=rgb.to(DEVICE),freq.to(DEVICE),geo.to(DEVICE)
            labels=labels.numpy()
            rgb_prob=torch.sigmoid(rgb_model(rgb))
            fft_prob=torch.sigmoid(fft_model(freq))
            geo_prob=torch.sigmoid(geo_model(geo))
            final_prob=0.5*rgb_prob+0.3*fft_prob+0.2*geo_prob
            preds=(final_prob>0.5).int().cpu().numpy()
            preds_all.extend(preds.flatten())
            labels_all.extend(labels.flatten())
    preds_all=np.array(preds_all)
    labels_all=np.array(labels_all)
    precision,recall,f1,acc,iou=compute_metrics(labels_all,preds_all)
    print("\n===== BOOSTED RESULT =====")
    print("Accuracy:",acc)
    print("Precision:",precision)
    print("Recall:",recall)
    print("F1:",f1)
    print("IoU:",iou)

ensemble_test(rgb_model,fft_model,geo_model,test_loader)
print("\nTraining Finished.")
