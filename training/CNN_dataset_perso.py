import os, glob
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf

DATA_DIR = "custom_digits"
TEST_RATIO = 0.1              # 10% test / 90% train
SEED = 42

"""
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
"""

# -----------------------------
# 1) Charger le dataset perso
# -----------------------------
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.bmp")))
if not files:
    raise FileNotFoundError(f"Aucun fichier .bmp trouvé dans {DATA_DIR}")

X, y = [], []

for f in files:
    base = os.path.basename(f)
    label = int(base.split("-")[0])

    img = Image.open(f).convert("L")   # grayscale (1 canal)
    if img.size != (28, 28):
        img = img.resize((28, 28))

    arr = np.array(img, dtype=np.float32) / 255.0

    X.append(arr)
    y.append(label)

x = np.array(X, dtype=np.float32)[..., np.newaxis]  # (N, 28, 28, 1)
y = np.array(y, dtype=np.int32)                     # (N,)

print("Dataset:", x.shape, y.shape, "labels uniques:", np.unique(y))

# -----------------------------
# 2) Split train/test (simple)
# -----------------------------
rng = np.random.default_rng(SEED)

indices = np.arange(len(x))
rng.shuffle(indices)

n_test = int(len(x) * TEST_RATIO)
test_idx = indices[:n_test]
train_idx = indices[n_test:]

x_train, y_train = x[train_idx], y[train_idx]
x_test,  y_test  = x[test_idx],  y[test_idx]

print("Train:", x_train.shape, y_train.shape)
print("Test :", x_test.shape, y_test.shape)


# -----------------------------
# 3) Modèle CNN
# -----------------------------
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(16, (5, 5), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (5, 5), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

# -----------------------------
# 4) Compilation (inchangé)
# -----------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# 5) Entraînement
# -----------------------------
history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=20,
    batch_size=16,          # batch plus petit aide souvent sur petit dataset
)

# -----------------------------
# 6) Évaluation
# -----------------------------
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

plt.tight_layout()
plt.savefig("predictions_CNN_data_perso_SANS_data_augmentation.png") 
print("predictions_CNN_data_perso_SANS_data_augmentation.png")


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
plt.savefig("training_curves_CNN.png")
plt.close()

print("Courbes sauvegardées dans training_curves_CNN.png")

