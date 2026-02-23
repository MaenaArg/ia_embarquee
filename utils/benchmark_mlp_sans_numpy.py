import os, glob, time, math
from PIL import Image
from tensorflow import keras

WEIGHTS_DIR = "mlp_weights_txt"
DATA_DIR    = "custom_digits"

# ------------------------------------------
# 1. Charger les poids depuis les .txt
#    (on utilise juste open/split, pas numpy)
# ------------------------------------------
def load_matrix(filepath):
    matrix = []
    with open(filepath) as f:
        for line in f:
            row = [float(x) for x in line.strip().split()]
            if row:
                matrix.append(row)
    return matrix

def load_vector(filepath):
    with open(filepath) as f:
        return [float(x) for x in f.read().split()]

w1 = load_matrix(os.path.join(WEIGHTS_DIR, "dense1_weights.txt"))  # (784, 128)
b1 = load_vector(os.path.join(WEIGHTS_DIR, "dense1_biases.txt"))   # (128,)
w2 = load_matrix(os.path.join(WEIGHTS_DIR, "dense2_weights.txt"))  # (128, 10)
b2 = load_vector(os.path.join(WEIGHTS_DIR, "dense2_biases.txt"))   # (10,)

print("Poids chargés.")

# ------------------------------------------
# 2. Forward pass SANS numpy - boucles pures
#    Identique au C
# ------------------------------------------
def forward(img):
    # Couche 1 : hidden = ReLU(W1^T * img + b1)
    hidden = [0.0] * 128
    for j in range(128):
        s = b1[j]
        for i in range(784):
            s += img[i] * w1[i][j]
        hidden[j] = s if s > 0.0 else 0.0   # ReLU

    # Couche 2 : output = W2^T * hidden + b2
    output = [0.0] * 10
    for k in range(10):
        s = b2[k]
        for j in range(128):
            s += hidden[j] * w2[j][k]
        output[k] = s

    # Softmax
    max_val = max(output)
    e = [math.exp(x - max_val) for x in output]
    s = sum(e)
    return [x / s for x in e]

# ------------------------------------------
# 3. Charger le dataset perso (sans numpy)
# ------------------------------------------
files  = sorted(glob.glob(os.path.join(DATA_DIR, "*.bmp")))
images = []
labels = []
for f in files:
    label = int(os.path.basename(f).split("-")[0])
    img   = list(Image.open(f).convert("L").resize((28, 28)).getdata())
    img   = [p / 255.0 for p in img]
    images.append(img)
    labels.append(label)

print(f"{len(images)} images perso chargées.")

# ------------------------------------------
# 4. Benchmark dataset perso
# ------------------------------------------
t0 = time.perf_counter()
preds_perso = [probs.index(max(probs)) for img in images for probs in [forward(img)]]
t1 = time.perf_counter()

time_perso_ms = (t1 - t0) / len(images) * 1000
acc_perso     = sum(p == l for p, l in zip(preds_perso, labels)) / len(labels)

# ------------------------------------------
# 5. Charger MNIST (sans numpy)
# ------------------------------------------
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
N_MNIST  = 1000
mnist_images = [[p / 255.0 for p in img.flatten().tolist()] for img in x_test[:N_MNIST]]
mnist_labels = y_test[:N_MNIST].tolist()

print(f"{N_MNIST} images MNIST chargées.")

# ------------------------------------------
# 6. Benchmark MNIST
# ------------------------------------------
t0 = time.perf_counter()
preds_mnist = [probs.index(max(probs)) for img in mnist_images for probs in [forward(img)]]
t1 = time.perf_counter()

time_mnist_ms = (t1 - t0) / N_MNIST * 1000
acc_mnist     = sum(p == l for p, l in zip(preds_mnist, mnist_labels)) / N_MNIST

# ------------------------------------------
# 7. Affichage
# ------------------------------------------
print("\n========================================")
print("     BENCHMARK MLP - Python pur")
print("========================================")
print(f"  Temps par image (dataset perso) : {time_perso_ms:.4f} ms")
print(f"  Temps par image (MNIST)         : {time_mnist_ms:.4f} ms")
print(f"  Accuracy dataset perso          : {acc_perso:.1%}")
print(f"  Accuracy MNIST                  : {acc_mnist:.1%}")
print("========================================")