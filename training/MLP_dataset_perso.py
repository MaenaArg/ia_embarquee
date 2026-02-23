import os, glob
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from scipy import ndimage

DATA_DIR = "custom_digits"
TEST_RATIO = 0.1

"""
SEED = 42

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
"""

# ------------------------------------------
# 1. Charger les images et extraire les labels
# ------------------------------------------
X = []
y = []

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.bmp")))
if not files:
    raise FileNotFoundError("Aucune image .bmp trouvée")

for f in files:
    base = os.path.basename(f)
    label = int(base.split("-")[0])

    img = Image.open(f).convert("L")  # grayscale
    img = img.resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0

    X.append(arr)
    y.append(label)

X = np.array(X)[..., np.newaxis]   # (N,28,28,1)
y = np.array(y, dtype=np.int32)

# ------------------------------------------
# 2. Split simple (shuffle + 80/20)
# ------------------------------------------
N = len(X)
indices = np.arange(N)
np.random.shuffle(indices)

n_test = int(N * TEST_RATIO)
test_idx = indices[:n_test]
train_idx = indices[n_test:]

x_train, y_train = X[train_idx], y[train_idx]
x_test,  y_test  = X[test_idx],  y[test_idx]

# ------------------------------------------
# 3. Modèle MLP identique à MNIST
# ------------------------------------------
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ------------------------------------------
# 4. Entraînement
# ------------------------------------------
history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=16,
    validation_split=0.1,
)

# ------------------------------------------
# 5. Évaluation
# ------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")


# Sélectionner 10 images aléatoires du dataset perso
indices = np.random.choice(len(x_test), 10, replace=False)

plt.figure(figsize=(12, 3))
for i, idx in enumerate(indices):
    img = x_test[idx].squeeze()  # (28,28)
    label_true = y_test[idx]
    label_pred = np.argmax(model.predict(x_test[idx:idx+1], verbose=0))

    plt.subplot(1, 10, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Valeur:{label_true}\nPrédiction:{label_pred}")
    plt.axis('off')

# Au lieu de plt.show(), on sauvegarde l'image
plt.tight_layout()
plt.savefig("predictions_MLP_data_perso_SANS_data_augmentation.png")
plt.close()
print("Image sauvegardée sous predictions_MLP_data_perso_SANS_data_augmentation.png")


plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
plt.close()

print("Courbes sauvegardées dans training_curves.png")

# ------------------------------------------
# 6. Exporter les poids et biais du MLP en TXT
# ------------------------------------------
"""
WEIGHTS_DIR = "mlp_weights_txt"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

for i, layer in enumerate(model.layers):
    if isinstance(layer, layers.Dense):
        weights, biases = layer.get_weights()
        
        # Sauvegarde les poids
        weights_path = os.path.join(WEIGHTS_DIR, f"layer{i}_weights.txt")
        np.savetxt(weights_path, weights, fmt="%.6f")
        
        # Sauvegarde les biais
        biases_path = os.path.join(WEIGHTS_DIR, f"layer{i}_biases.txt")
        np.savetxt(biases_path, biases.reshape(1, -1), fmt="%.6f")
        
        print(f"Poids de la couche {i} sauvegardés dans {weights_path}")
        print(f"Biais de la couche {i} sauvegardés dans {biases_path}")

print(f"Tous les poids et biais sauvegardés dans le dossier {WEIGHTS_DIR}")
"""

