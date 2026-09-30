import pandas as pd
import numpy as np
from scipy.signal import butter, iirnotch, filtfilt
import matplotlib.pyplot as plt
import pywt
import glob
import os

FS = 1000

def preprocess(x, fs=FS):
    bn, an = iirnotch(60, 30, fs)
    x = filtfilt(bn, an, x, axis=0)

    b, a = butter(4, [20/(fs/2), 499/(fs/2)], btype='band')
    return filtfilt(b, a, x, axis=0)


file_path = "palm-sEMG-doorknob-filtered-main/data/A/a (1).csv"

df = pd.read_csv(file_path)
x = df.values.astype(float)

xf = preprocess(x)

print("원본 배열:", x.shape)
print("전:", x.std())
print("후:", xf.std())


from scipy.signal import welch

f0, P0 = welch(x[:, 0], fs=FS, nperseg=512)
f1, P1 = welch(xf[:, 0], fs=FS, nperseg=512)

plt.figure(figsize=(9, 3))
plt.semilogy(f0, P0, label='before', lw=0.8)
plt.semilogy(f1, P1, label='after', lw=0.8)
plt.axvline(60, color='r', ls='--')
plt.xlim(0, 300)
plt.legend()
plt.xlabel('Hz')
plt.ylabel('power')
plt.tight_layout()
plt.savefig('filter_check.png', dpi=120)
plt.close()


WIN, HOP = 300, 150

def make_windows(x, win=WIN, hop=HOP):
    n = (len(x) - win) // hop + 1

    return np.stack([
        x[i*hop:i*hop+win]
        for i in range(n)
    ])


w = make_windows(xf)

print("윈도우 배열 모양:", w.shape)


def minmax(w, eps=1e-8):
    mn = w.min(axis=(1, 2), keepdims=True)
    mx = w.max(axis=(1, 2), keepdims=True)

    return (w - mn) / (mx - mn + eps)


wn = minmax(w)

print("값 범위:", wn.min(), "~", wn.max())


SCALES = np.arange(1, 33)

def to_cwt(one_window, wavelet='morl'):
    maps = []

    for ch in range(one_window.shape[1]):
        coef, _ = pywt.cwt(
            one_window[:, ch],
            SCALES,
            wavelet
        )

        maps.append(np.abs(coef))

    maps.append((maps[0] + maps[1]) / 2)

    return np.stack(maps).astype(np.float32)


t = to_cwt(wn[0])

print("입력 텐서 모양:", t.shape)


plt.figure(figsize=(6, 4))
plt.imshow(t[0], aspect='auto', origin='lower')
plt.xlabel('Time')
plt.ylabel('Scale')
plt.tight_layout()
plt.savefig('cwt_check.png', dpi=120)
plt.close()


from sklearn.model_selection import train_test_split

files = glob.glob(
    "palm-sEMG-doorknob-filtered-main/data/*/*.csv"
)

labels = [
    os.path.basename(os.path.dirname(f))
    for f in files
]

tr_files, te_files = train_test_split(
    files,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

print("전체 시행:", len(files))
print("학습 시행:", len(tr_files))
print("테스트 시행:", len(te_files))


def load(f):
    return pd.read_csv(f).values.astype(float)


def label_of(f):
    return os.path.basename(os.path.dirname(f))


def build(flist):
    X, y = [], []

    for f in flist:
        sig = preprocess(load(f))

        for window in minmax(make_windows(sig)):
            X.append(to_cwt(window))
            y.append(label_of(f))

    return np.stack(X), np.array(y)


Xtr, ytr = build(tr_files)
Xte, yte = build(te_files)

print("학습 데이터:", Xtr.shape)
print("학습 라벨:", ytr.shape)
print("테스트 데이터:", Xte.shape)
print("테스트 라벨:", yte.shape)


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import densenet161, resnet18


label_map = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4
}

ytr_num = np.array([
    label_map[y]
    for y in ytr
])

yte_num = np.array([
    label_map[y]
    for y in yte
])


train_ds = TensorDataset(
    torch.tensor(Xtr, dtype=torch.float32),
    torch.tensor(ytr_num, dtype=torch.long)
)

test_ds = TensorDataset(
    torch.tensor(Xte, dtype=torch.float32),
    torch.tensor(yte_num, dtype=torch.long)
)


train_loader = DataLoader(
    train_ds,
    batch_size=4,
    shuffle=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=16,
    shuffle=False
)


dev = 'cuda' if torch.cuda.is_available() else 'cpu'

print("사용 장치:", dev)


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


classes = ['A', 'B', 'C', 'D', 'E']


def evaluate(model, name):

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for xb, yb in test_loader:

            xb = xb.to(dev)

            outputs = model(xb)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            y_true.extend(
                yb.numpy()
            )

            y_pred.extend(
                predictions.cpu().numpy()
            )


    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average='macro',
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average='macro',
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average='macro',
        zero_division=0
    )


    print()
    print("================================")
    print(name)
    print("================================")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)


    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4]
    )

    print()
    print(name, "Confusion Matrix:")
    print(cm)


    plt.figure(figsize=(6, 5))

    plt.imshow(cm)

    plt.title(name + ' Confusion Matrix')

    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.xticks(
        range(5),
        classes
    )

    plt.yticks(
        range(5),
        classes
    )


    for i in range(5):

        for j in range(5):

            plt.text(
                j,
                i,
                cm[i, j],
                ha='center',
                va='center'
            )


    plt.tight_layout()

    filename = name.lower().replace(' ', '_') + '_confusion_matrix.png'

    plt.savefig(
        filename,
        dpi=150
    )

    plt.close()

    return accuracy, precision, recall, f1


def train_model(model, name, epochs=3):

    model = model.to(dev)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    criterion = nn.CrossEntropyLoss()


    print()
    print("================================")
    print(name, "학습 시작")
    print("================================")


    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for xb, yb in train_loader:

            xb = xb.to(dev)
            yb = yb.to(dev)

            outputs = model(xb)

            loss = criterion(
                outputs,
                yb
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()


        avg_loss = total_loss / len(train_loader)

        print(
            name,
            "Epoch:",
            epoch + 1,
            "/",
            epochs,
            "Loss:",
            avg_loss
        )


    return model


model1 = densenet161(
    weights=None
)

model1.classifier = nn.Linear(
    2208,
    5
)


model1 = train_model(
    model1,
    "DenseNet161",
    epochs=3
)


dense_result = evaluate(
    model1,
    "DenseNet161"
)


model2 = resnet18(
    weights=None
)

model2.fc = nn.Linear(
    512,
    5
)


model2 = train_model(
    model2,
    "ResNet18",
    epochs=3
)


resnet_result = evaluate(
    model2,
    "ResNet18"
)


print()
print("================================")
print("최종 모델 성능 비교")
print("================================")

print(
    "DenseNet161:",
    dense_result
)

print(
    "ResNet18:",
    resnet_result
)