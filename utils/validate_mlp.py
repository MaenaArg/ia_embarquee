import os, glob
import numpy as np
from PIL import Image

WEIGHTS_DIR = "mlp_weights_txt"
DATA_DIR    = "custom_digits"
OUTPUT_FILE = "predictions_python.txt"

# ------------------------------------------
# 1. Charger les poids depuis les .txt
# ------------------------------------------
w1 = np.loadtxt(os.path.join(WEIGHTS_DIR, "dense1_weights.txt"), dtype=np.float32)  # (784, 128)
b1 = np.loadtxt(os.path.join(WEIGHTS_DIR, "dense1_biases.txt"),  dtype=np.float32)  # (128,)
w2 = np.loadtxt(os.path.join(WEIGHTS_DIR, "dense2_weights.txt"), dtype=np.float32)  # (128, 10)
b2 = np.loadtxt(os.path.join(WEIGHTS_DIR, "dense2_biases.txt"),  dtype=np.float32)  # (10,)

print(f"w1 : {w1.shape}  b1 : {b1.shape}")
print(f"w2 : {w2.shape}  b2 : {b2.shape}")

# ------------------------------------------
# 2. Forward pass manuel (identique au C)
# ------------------------------------------
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x))   # stabilite numerique
    return e / e.sum()

def forward(img_flat):
    hidden = relu(w1.T @ img_flat + b1)   # (128,)
    output = softmax(w2.T @ hidden + b2)  # (10,)
    return output

# ------------------------------------------
# 3. Charger toutes les images du dataset perso
# ------------------------------------------
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.bmp")))
if not files:
    raise FileNotFoundError("Aucune image .bmp trouvee dans custom_digits/")

print(f"\n{len(files)} images trouvees\n")

# ------------------------------------------
# 4. Faire les predictions et sauvegarder
# ------------------------------------------
results = []

with open(OUTPUT_FILE, "w") as f_out:
    f_out.write(f"{'Image':<30} {'Label':>5} {'Prediction':>10} {'Correct':>8}\n")
    f_out.write("-" * 60 + "\n")

    for filepath in files:
        basename = os.path.basename(filepath)
        label    = int(basename.split("-")[0])

        img = Image.open(filepath).convert("L").resize((28, 28))
        arr = np.array(img, dtype=np.float32).flatten() / 255.0

        probs      = forward(arr)
        prediction = int(np.argmax(probs))
        correct    = "OK" if prediction == label else "ERREUR"

        results.append((basename, label, prediction, correct))
        f_out.write(f"{basename:<30} {label:>5} {prediction:>10} {correct:>8}\n")
print(f"Prédictions pyhton sauvegardée dans le fichier {OUTPUT_FILE}")