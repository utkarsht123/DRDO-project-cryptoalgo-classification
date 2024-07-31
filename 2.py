import numpy as np
from Crypto.Cipher import DES3, AES
from Crypto.Random import get_random_bytes
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_all = np.concatenate((x_train, x_test))
y_all = np.concatenate((y_train, y_test))

# Split data for TDES and AES encryption
tdes_subset = x_all[:30000]
aes_subset = x_all[30000:60000]

# Encryption functions
def encrypt_image_tdes(image, key):
    cipher = DES3.new(key, DES3.MODE_ECB)
    return cipher.encrypt(image.tobytes())

def encrypt_image_aes(image, key):
    cipher = AES.new(key, AES.MODE_ECB)
    return cipher.encrypt(image.tobytes())

# Generate random keys for encryption
tdes_key = DES3.adjust_key_parity(get_random_bytes(24))
aes_key = get_random_bytes(16)

# Encrypt the images
tdes_encrypted = [encrypt_image_tdes(img, tdes_key) for img in tdes_subset]
aes_encrypted = [encrypt_image_aes(img, aes_key) for img in aes_subset]

# Convert encrypted data to numpy arrays
tdes_dataset = np.array([np.frombuffer(img, dtype=np.uint8).reshape(28, 28) for img in tdes_encrypted])
aes_dataset = np.array([np.frombuffer(img, dtype=np.uint8).reshape(28, 28) for img in aes_encrypted])

# Save the encrypted datasets
np.save('tdes_encrypted_mnist.npy', tdes_dataset)
np.save('aes_encrypted_mnist.npy', aes_dataset)

# Load the encrypted datasets
aes_dataset = np.load('aes_encrypted_mnist.npy')
tdes_dataset = np.load('tdes_encrypted_mnist.npy')

# Combine AES and TDES datasets
encrypted_mnist = {
    'images': np.concatenate((aes_dataset, tdes_dataset)),
    'labels': np.concatenate((np.zeros(len(aes_dataset)), np.ones(len(tdes_dataset))))
}
np.save('EncryptedMNIST_AES_TDES.npy', encrypted_mnist)

# Load combined dataset
encrypted_mnist = np.load('EncryptedMNIST_AES_TDES.npy', allow_pickle=True).item()
X = encrypted_mnist['images']
y = encrypted_mnist['labels']

# Normalize and reshape data
X = X.astype('float32') / 255.0
X = X.reshape(X.shape[0], 28, 28, 1)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data generator function
def data_generator(X, y, batch_size):
    while True:
        idx = np.random.randint(0, X.shape[0], batch_size)
        yield X[idx], y[idx]

# Preprocess image for MobileNetV2
def preprocess_image(image):
    image = tf.image.resize(image, (96, 96))
    image = tf.image.grayscale_to_rgb(image)
    return image

# Load MobileNetV2 and freeze the base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
base_model.trainable = False

# Define the model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Lambda(preprocess_image),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training parameters
batch_size = 64
train_generator = data_generator(X_train, y_train, batch_size)
validation_generator = data_generator(X_test, y_test, batch_size)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(X_test) // batch_size
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Function to test the model on a few images
def test_on_images(model, X, y, num_images=10):
    indices = np.random.choice(len(X), num_images, replace=False)
    plt.figure(figsize=(15, 3 * num_images))
    for i, idx in enumerate(indices):
        image = X[idx]
        true_label = y[idx]
        prediction = model.predict(np.expand_dims(image, axis=0))[0][0]
        predicted_label = int(prediction > 0.5)
        
        plt.subplot(num_images, 2, 2 * i + 1)
        plt.imshow(image.reshape(28, 28), cmap='gray')
        plt.title(f"True: {'TDES' if true_label else 'AES'}")
        plt.axis('off')

        plt.subplot(num_images, 2, 2 * i + 2)
        plt.bar(['AES', 'TDES'], [1 - prediction, prediction])
        plt.title(f"Predicted: {'TDES' if predicted_label else 'AES'}")
        plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

# Test the model on some images
test_on_images(model, X_test, y_test, num_images=10)
