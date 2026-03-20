import sys
import os
import cv2
import torch
import mediapipe as mp
import numpy as np
import torch.nn as nn
from torchvision import models, transforms
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt

# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = r"D:\proje klasörleri\CV_DeepFake\outputs\best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# MEDIAPIPE FACEMESH
# =====================================================

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def extract_geometry(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return np.zeros(5)

    landmarks = results.multi_face_landmarks[0].landmark

    def dist(p1,p2):
        return np.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)

    eye_dist = dist(landmarks[33],landmarks[263])
    mouth_width = dist(landmarks[61],landmarks[291])
    nose_chin = dist(landmarks[1],landmarks[152])
    face_width = dist(landmarks[234],landmarks[454])
    eye_width = dist(landmarks[33],landmarks[133])

    return np.array([
        eye_dist/face_width,
        mouth_width/face_width,
        nose_chin/face_width,
        eye_width/face_width,
        face_width
    ])

# =====================================================
# FFT
# =====================================================

def compute_fft(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20*np.log(np.abs(fshift)+1)
    magnitude = cv2.resize(magnitude,(224,224))
    magnitude = np.stack([magnitude]*3,axis=-1)
    return magnitude.astype(np.uint8)

# =====================================================
# MODEL (ExplainableHybrid ile birebir aynı)
# =====================================================

class ExplainableHybrid(nn.Module):
    def __init__(self):
        super().__init__()

        self.rgb=models.resnet34(weights=None)
        self.rgb.fc=nn.Identity()

        self.freq=models.resnet18(weights=None)
        self.freq.fc=nn.Identity()

        self.geo_branch=nn.Sequential(
            nn.Linear(5,32),
            nn.ReLU(),
            nn.Linear(32,64)
        )

        self.rgb_head=nn.Linear(512,1)
        self.freq_head=nn.Linear(512,1)
        self.geo_head=nn.Linear(64,1)

        self.fusion=nn.Sequential(
            nn.Linear(512+512+64,512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512,1)
        )

    def forward(self,rgb,freq,geo):
        f_rgb=self.rgb(rgb)
        f_freq=self.freq(freq)
        f_geo=self.geo_branch(geo)

        rgb_logit=self.rgb_head(f_rgb)
        freq_logit=self.freq_head(f_freq)
        geo_logit=self.geo_head(f_geo)

        fused=torch.cat([f_rgb,f_freq,f_geo],dim=1)
        final_logit=self.fusion(fused)

        return final_logit,rgb_logit,freq_logit,geo_logit

# =====================================================
# GUI
# =====================================================

class DeepfakeApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Explainable Deepfake Detector (ResNet34+18+FaceMesh)")
        self.setGeometry(200,100,700,800)
        self.setStyleSheet("background-color:#1e1e1e;color:white;")

        self.model = ExplainableHybrid().to(DEVICE)
        self.model.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE))
        self.model.eval()

        self.init_ui()

    def init_ui(self):

        layout=QVBoxLayout()

        self.title=QLabel("Explainable Deepfake Detection")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setFont(QFont("Arial",18,QFont.Bold))
        self.title.setStyleSheet("color:#00d2ff;margin:15px;")
        layout.addWidget(self.title)

        self.image_label=QLabel("Image Preview")
        self.image_label.setFixedSize(500,400)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border:2px dashed #444;")
        layout.addWidget(self.image_label,alignment=Qt.AlignCenter)

        self.result_label=QLabel("Result: Waiting...")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial",14))
        layout.addWidget(self.result_label)

        self.branch_label=QLabel("")
        self.branch_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.branch_label)

        btn=QPushButton("Select Image")
        btn.clicked.connect(self.load_image)
        btn.setStyleSheet("background-color:#007bff;padding:10px;")
        layout.addWidget(btn)

        self.setLayout(layout)

    def preprocess(self,img_path):

        img=cv2.imread(img_path)
        img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])

        rgb_tensor=transform(img_rgb).unsqueeze(0).to(DEVICE)

        fft_img=compute_fft(img)
        freq_tensor=transform(fft_img).unsqueeze(0).to(DEVICE)

        geo=torch.tensor(extract_geometry(img),
                         dtype=torch.float32).unsqueeze(0).to(DEVICE)

        return img_rgb,rgb_tensor,freq_tensor,geo

    def load_image(self):

        file,_=QFileDialog.getOpenFileName(self,"Select Image","","Images (*.png *.jpg *.jpeg)")

        if not file:
            return

        img_rgb,rgb,freq,geo=self.preprocess(file)

        with torch.no_grad():
            final_logit,rgb_l,freq_l,geo_l=self.model(rgb,freq,geo)

            prob=torch.sigmoid(final_logit).item()
            rgb_score=torch.sigmoid(rgb_l).item()
            freq_score=torch.sigmoid(freq_l).item()
            geo_score=torch.sigmoid(geo_l).item()

        total=rgb_score+freq_score+geo_score+1e-6
        rgb_c=rgb_score/total
        freq_c=freq_score/total
        geo_c=geo_score/total

        is_fake=prob>0.5
        percent=prob*100 if is_fake else (1-prob)*100

        if is_fake:
            self.result_label.setText(f"FAKE - Confidence: %{percent:.2f}")
            self.result_label.setStyleSheet("color:red;font-weight:bold;")
        else:
            self.result_label.setText(f"REAL - Confidence: %{percent:.2f}")
            self.result_label.setStyleSheet("color:lime;font-weight:bold;")

        self.branch_label.setText(
            f"Spatial: {rgb_c:.2f} | Frequency: {freq_c:.2f} | Geometry: {geo_c:.2f}"
        )

        self.display_image(img_rgb)

    def display_image(self,img):
        h,w,ch=img.shape
        bytes_per_line=ch*w
        q_img=QImage(img.data,w,h,bytes_per_line,QImage.Format_RGB888)
        pixmap=QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(
            500,400,Qt.KeepAspectRatio,Qt.SmoothTransformation))

# =====================================================
# RUN
# =====================================================

if __name__=="__main__":
    app=QApplication(sys.argv)
    window=DeepfakeApp()
    window.show()
    sys.exit(app.exec_())
