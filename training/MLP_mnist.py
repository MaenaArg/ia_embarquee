from tensorflow import keras               # Importe l’API Keras intégrée à TensorFlow
from keras import layers                  # Importe le module des couches (Dense, Dropout, etc.)
from keras.datasets import mnist          # Importe le dataset MNIST (chiffres 0-9)
import matplotlib.pyplot as plt
import numpy as np

# Charger et préparer les données
(x_train, y_train), (x_test, y_test) = mnist.load_data()  

x_train, x_test = x_train / 255.0, x_test / 255.0  
# Normalise les pixels de [0..255] vers [0..1] pour faciliter l’apprentissage (plus stable)

# Définir le modèle avec l’API Sequential (pipeline de couches empilées)
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),     
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.summary()  # Affiche l’architecture (couches + nombre de paramètres)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=20,                   # Nombre de passes complètes sur le dataset
    batch_size=16,                # Taille des mini-lots (plus petit → plus de mises à jour)
    validation_split=0.1,        # 10% des données train servent à valider pendant le training
)

# Évaluer sur le test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
# evaluate calcule loss + accuracy sur les données de test (jamais vues à l’entraînement)

print(f"Test accuracy: {test_acc:.4f}")
# Affiche l’accuracy finale sur test

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
plt.savefig("predictions_MLP_MNIST.png")  # sera créée dans le dossier courant
plt.close()
print("Image sauvegardée sous predictions_MLP_MNIST.png")
