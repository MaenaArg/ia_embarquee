import os, glob, time
import numpy as np
from PIL import Image
from tensorflow import keras

WEIGHTS_DIR = "mlp_weights_txt"
DATA_DIR    = "custom_digits"
N_REPEAT    = 10

# Charger les poids
w1 = np.loadtxt(os.path.join(WEIGHTS_DIR, "dense1_weights.txt"), dtype=np.float32)
b1 = np.loadtxt(os.path.join(WEIGHTS_DIR, "dense1_biases.txt"),  dtype=np.float32)
w2 = np.loadtxt(os.path.join(WEIGHTS_DIR, "dense2_weights.txt"), dtype=np.float32)
b2 = np.loadtxt(os.path.join(WEIGHTS_DIR, "dense2_biases.txt"),  dtype=np.float32)

# Forward pass Python pur (identique au C)
def forward(img_flat):
    hidden = np.maximum(0, w1.T @ img_flat + b1)
    output = w2.T @ hidden + b2
    e = np.exp(output - np.max(output))
    return e / e.sum()

# Charger dataset perso
files, images, labels = sorted(glob.glob(os.path.join(DATA_DIR, "*.bmp"))), [], []
for f in files:
    label = int(os.path.basename(f).split("-")[0])
    img   = Image.open(f).convert("L").resize((28, 28))
    images.append(np.array(img, dtype=np.float32).flatten() / 255.0)
    labels.append(label)
images, labels = np.array(images), np.array(labels)

# Benchmark dataset perso
t0 = time.perf_counter()
for _ in range(N_REPEAT):
    preds_perso = [np.argmax(forward(img)) for img in images]
t1 = time.perf_counter()
time_perso_ms = (t1 - t0) / N_REPEAT / len(images) * 1000
acc_perso     = np.mean(np.array(preds_perso) == labels)

# Benchmark MNIST (1000 images)
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = x_test.astype(np.float32).reshape(-1, 784) / 255.0
N_MNIST = 1000
t0 = time.perf_counter()
preds_mnist = [np.argmax(forward(img)) for img in x_test[:N_MNIST]]
t1 = time.perf_counter()
time_mnist_ms = (t1 - t0) / N_MNIST * 1000
acc_mnist     = np.mean(np.array(preds_mnist) == y_test[:N_MNIST])

# Affichage
print("\n========================================")
print("        BENCHMARK MLP - Python")
print("========================================")
print(f"  Temps par image (dataset perso) : {time_perso_ms:.2f} ms")
print(f"  Temps par image (MNIST)         : {time_mnist_ms:.2f} ms")
print(f"  Accuracy dataset perso          : {acc_perso:.1%}")
print(f"  Accuracy MNIST                  : {acc_mnist:.1%}")
print("========================================")