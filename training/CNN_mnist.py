import tensorflow as tf
from tensorflow.keras import layers, models
from keras.datasets import mnist  
import matplotlib.pyplot as plt
import numpy as np

# Charger et préparer les données
(x_train, y_train), (x_test, y_test) = mnist.load_data() 

# normalisation (inchangé) + reshape pour CNN (N, 28, 28, 1)
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

x_train = x_train[..., tf.newaxis]  # (60000, 28, 28, 1)
x_test  = x_test[..., tf.newaxis]   # (10000, 28, 28, 1)

# 3) Modèle CNN
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

# 4) Compilation (comme MLP, mais CNN)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",  # labels entiers (0..9)
    metrics=["accuracy"]
)

# 5) Entraînement
history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=5,
    batch_size=128
)

# 6) Évaluation
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
plt.savefig("predictions_CNN_MNIST.png")  # sera créée dans le dossier courant
plt.close()
print("Image sauvegardée sous predictions_CNN_MNIST.png")
