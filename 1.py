import numpy as np
from Crypto.Cipher import DES3, AES
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# Encrypt image using TDES
def encrypt_tdes(image, key):
    cipher = DES3.new(key, DES3.MODE_ECB)
    padding_length = 8 - (image.size % 8) if image.size % 8 != 0 else 0
    padded_image = np.pad(image.flatten(), (0, padding_length), 'constant')
    encrypted_image = cipher.encrypt(padded_image.tobytes())
    return np.frombuffer(encrypted_image, dtype=np.uint8).reshape(image.shape)

# Encrypt image using AES
def encrypt_aes(image, key):
    cipher = AES.new(key, AES.MODE_ECB)
    padding_length = 16 - (image.size % 16) if image.size % 16 != 0 else 0
    padded_image = np.pad(image.flatten(), (0, padding_length), 'constant')
    encrypted_image = cipher.encrypt(padded_image.tobytes())
    new_shape = (image.shape[0], image.shape[1], image.shape[2] + padding_length // image.shape[1])
    return np.frombuffer(encrypted_image, dtype=np.uint8).reshape(new_shape)

# Define keys
tdes_key = b'Sixteen byte key'
aes_key = b'Sixteen byte key'

# Example data (x_train should be loaded with actual image data)
# x_train = np.array([...])  # Load your image data here

# Encrypt the training data with both TDES and AES
x_train_tdes = np.array([encrypt_tdes(img, tdes_key) for img in x_train])
x_train_aes = np.array([encrypt_aes(img, aes_key) for img in x_train])

# Create labels for the encrypted data
y_train_tdes = np.zeros(x_train_tdes.shape[0])  # Label 0 for TDES
y_train_aes = np.ones(x_train_aes.shape[0])     # Label 1 for AES

# Combine the encrypted data and labels
x_train_combined = np.concatenate((x_train_tdes, x_train_aes), axis=0)
y_train_combined = np.concatenate((y_train_tdes, y_train_aes), axis=0)

# Split the data into training and validation sets
x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
    x_train_combined, y_train_combined, test_size=0.2, random_state=42
)

# Normalize the data
x_train_split = x_train_split / 255.0
x_val_split = x_val_split / 255.0

# One-hot encode the labels
y_train_split = to_categorical(y_train_split, 2)
y_val_split = to_categorical(y_val_split, 2)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train_split, y_train_split, epochs=10, validation_data=(x_val_split, y_val_split))

# Save the model
model.save('encryption_classifier_model.h5')

# Define a function to predict the type of encryption
def predict_encryption_type(image, model):
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return np.argmax(prediction)

# Load the model for testing
loaded_model = tf.keras.models.load_model('encryption_classifier_model.h5')

# Example data for testing (x_test should be loaded with actual image data)
# x_test = np.array([...])  # Load your test image data here

# Example of encrypted images for prediction
example_image = x_test[0]  # Use an example image from the test set
encrypted_example_tdes = encrypt_tdes(example_image, tdes_key)
encrypted_example_aes = encrypt_aes(example_image, aes_key)

# Predictions
print("TDES encrypted image prediction:", predict_encryption_type(encrypted_example_tdes, loaded_model))
print("AES encrypted image prediction:", predict_encryption_type(encrypted_example_aes, loaded_model))
