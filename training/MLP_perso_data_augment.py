import os, glob
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

DATA_DIR = "custom_digits"
TEST_RATIO = 0.2  # 20% du dataset perso pour le test
WEIGHTS_DIR = "mlp_weights_txt"

# ------------------------------------------
# 1. Charger MNIST
# ------------------------------------------
(x_mnist_train, y_mnist_train), (x_mnist_test, y_mnist_test) = keras.datasets.mnist.load_data()

x_mnist_train = x_mnist_train.astype(np.float32) / 255.0
x_mnist_train = x_mnist_train[..., np.newaxis]  # (60000, 28, 28, 1)

x_mnist_test = x_mnist_test.astype(np.float32) / 255.0
x_mnist_test = x_mnist_test[..., np.newaxis]

# ------------------------------------------
# 2. Charger le dataset perso
# ------------------------------------------
X = []
y = []

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.bmp")))
if not files:
    raise FileNotFoundError("Aucune image .bmp trouvée")

for f in files:
    base = os.path.basename(f)
    label = int(base.split("-")[0])

    img = Image.open(f).convert("L")
    img = img.resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0

    X.append(arr)
    y.append(label)

X = np.array(X)[..., np.newaxis]
y = np.array(y, dtype=np.int32)

# Split du dataset perso
N = len(X)
indices = np.arange(N)
np.random.shuffle(indices)

n_test = int(N * TEST_RATIO)
test_idx  = indices[:n_test]
train_idx = indices[n_test:]

x_perso_train, y_perso_train = X[train_idx], y[train_idx]
x_perso_test,  y_perso_test  = X[test_idx],  y[test_idx]

print(f"Dataset perso : {len(x_perso_train)} train, {len(x_perso_test)} test")

# ------------------------------------------
# 3. Mélanger MNIST + dataset perso pour l'entraînement
# ------------------------------------------
# On répète le dataset perso pour qu'il ait plus de poids
REPEAT = 50  # répéter les images perso 50x pour équilibrer avec MNIST
x_perso_repeated = np.repeat(x_perso_train, REPEAT, axis=0)
y_perso_repeated = np.repeat(y_perso_train, REPEAT, axis=0)

x_train_mix = np.concatenate([x_mnist_train, x_perso_repeated], axis=0)
y_train_mix = np.concatenate([y_mnist_train, y_perso_repeated], axis=0)

# Mélanger aléatoirement
shuffle_idx = np.random.permutation(len(x_train_mix))
x_train_mix = x_train_mix[shuffle_idx]
y_train_mix = y_train_mix[shuffle_idx]

print(f"Dataset d'entraînement total : {len(x_train_mix)} images")

# ------------------------------------------
# 4. Data Augmentation (légère)
# ------------------------------------------
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomZoom(0.08),
])

# ------------------------------------------
# 5. Modèle MLP
# ------------------------------------------
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    data_augmentation,
    layers.Flatten(),
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
# 6. Visualisation des données augmentées
# ------------------------------------------
plt.figure(figsize=(12, 4))
nb_exemples = 5

for i in range(nb_exemples):
    img_originale = x_perso_train[i]  # image originale

    # Appliquer l'augmentation (on appelle le modèle d'augmentation directement)
    img_augmentee = data_augmentation(img_originale[np.newaxis, ...], training=True)[0]

    # Colonne de gauche : originale
    plt.subplot(2, nb_exemples, i + 1)
    plt.imshow(img_originale.squeeze(), cmap='gray')
    plt.title(f"Original\n(label: {y_perso_train[i]})")
    plt.axis('off')

    # Colonne de droite : augmentée
    plt.subplot(2, nb_exemples, nb_exemples + i + 1)
    plt.imshow(img_augmentee.numpy().squeeze(), cmap='gray')
    plt.title("Augmentée")
    plt.axis('off')

plt.suptitle("Exemples de data augmentation", fontsize=13)
plt.tight_layout()
plt.savefig("exemples_data_augmentation_MLP.png")
plt.close()
print("Exemples d'augmentation sauvegardés dans exemples_data_augmentation_MLP.png")

# ------------------------------------------
# 7. Entraînement sur le mix
# ------------------------------------------
history = model.fit(
    x_train_mix, y_train_mix,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
)

# ------------------------------------------
# 8. Évaluation
# ------------------------------------------
test_loss, test_acc = model.evaluate(x_perso_test, y_perso_test, verbose=0)
print(f"Test accuracy sur dataset perso : {test_acc:.4f}")

mnist_loss, mnist_acc = model.evaluate(x_mnist_test, y_mnist_test, verbose=0)
print(f"Test accuracy sur MNIST         : {mnist_acc:.4f}")

# ------------------------------------------
# 9. Affichage des prédictions sur dataset perso
# ------------------------------------------
indices_plot = np.random.choice(len(x_perso_test), min(10, len(x_perso_test)), replace=False)

plt.figure(figsize=(12, 3))
for i, idx in enumerate(indices_plot):
    img = x_perso_test[idx].squeeze()
    label_true = y_perso_test[idx]
    label_pred = np.argmax(model.predict(x_perso_test[idx:idx+1], verbose=0))

    plt.subplot(1, len(indices_plot), i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Valeur:{label_true}\nPrédiction:{label_pred}")
    plt.axis('off')

plt.tight_layout()
plt.savefig("predictions_MLP_data_perso_AVEC_data_augmentation.png")
plt.close()
print("Image sauvegardée sous predictions_MLP_data_perso_AVEC_data_augmentation.png")

# ------------------------------------------
# 10. Courbes d'apprentissage
# ------------------------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves_augmentation_MLP.png")
plt.close()
print("Courbes sauvegardées dans training_curves_augmentation_MLP.png")

# ------------------------------------------
# 11. Matrice de confusion
# ------------------------------------------
y_pred = np.argmax(model.predict(x_perso_test, verbose=0), axis=1)
cm = confusion_matrix(y_perso_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10),
            cbar_kws={'label': 'Nombre de prédictions'})
plt.title(f'Matrice de confusion - MLP\nAccuracy: {test_acc:.1%}', fontsize=14, fontweight='bold')
plt.ylabel('Vraie valeur', fontsize=12)
plt.xlabel('Prédiction', fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix_MLP.png", dpi=120)
plt.close()
print("Matrice de confusion sauvegardée dans confusion_matrix_MLP.png")

# ------------------------------------------
# 12. Export des poids en .txt
# ------------------------------------------
WEIGHTS_DIR = "mlp_weights_txt"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# On cherche les couches Dense par leur type (plus robuste que par index)
dense_layers = [l for l in model.layers if isinstance(l, layers.Dense)]
print(f"Couches Dense trouvees : {[l.name for l in dense_layers]}")

dense1 = dense_layers[0]   # Dense(128)
w1, b1 = dense1.get_weights()
np.savetxt(os.path.join(WEIGHTS_DIR, "dense1_weights.txt"), w1, fmt="%.6f")
np.savetxt(os.path.join(WEIGHTS_DIR, "dense1_biases.txt"),  b1.reshape(1, -1), fmt="%.6f")
print(f"Dense1 - poids : {w1.shape}, biais : {b1.shape}")

dense2 = dense_layers[1]   # Dense(10)
w2, b2 = dense2.get_weights()
np.savetxt(os.path.join(WEIGHTS_DIR, "dense2_weights.txt"), w2, fmt="%.6f")
np.savetxt(os.path.join(WEIGHTS_DIR, "dense2_biases.txt"),  b2.reshape(1, -1), fmt="%.6f")
print(f"Dense2 - poids : {w2.shape}, biais : {b2.shape}")

print(f"Poids exportes dans le dossier : {WEIGHTS_DIR}/")
print("  dense1_weights.txt  (784 x 128)")
print("  dense1_biases.txt   (1 x 128)")
print("  dense2_weights.txt  (128 x 10)")
print("  dense2_biases.txt   (1 x 10)")
