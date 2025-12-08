import cv2
import numpy as np
import joblib
import onnxruntime as ort
import torch
from torchvision import transforms

W_SVM = 0.1
W_MLP = 0.1
W_RESNET = 0.8

class EnsembleModel:
    def __init__(self):
        self.svm = joblib.load("ensemble/svm_landmark_mediapipe.pkl")
        self.mlp = joblib.load("ensemble/mlp_landmark_mediapipe.pkl")

        self.session = ort.InferenceSession(
            "ensemble/resnet50_ensemble_mediapipe.onnx",
            providers=["CPUExecutionProvider"]
        )

        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

    def predict_resnet(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = self.tf(rgb).unsqueeze(0).numpy()
        logits = self.session.run(["logits"], {"input": x})[0]
        prob = 1 / (1 + np.exp(-logits))
        return float(prob)

    def predict(self, img_bgr, mediapipe_feat):
        mediapipe_feat = mediapipe_feat.reshape(1, -1)

        svm_p = float(self.svm.predict_proba(mediapipe_feat)[0,1])
        mlp_p = float(self.mlp.predict_proba(mediapipe_feat)[0,1])
        res_p = self.predict_resnet(img_bgr)

        final = (W_SVM*svm_p + W_MLP*mlp_p + W_RESNET*res_p) / (W_SVM+W_MLP+W_RESNET)

        return {
            "svm": svm_p,
            "mlp": mlp_p,
            "resnet": res_p,
            "final_prob": final,
            "label": "fake" if final > 0.5 else "real"
        }

