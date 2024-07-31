Problem statement
Crypto Algorithms Identification using ML
Title: Cryptographic Algorithm Identification using Machine Learning
1. Aim: The primary aim of this study is to develop a machine learning model capable of
identifying which cryptographic algorithm (AES or TDES) was used to encrypt an
image, using only the encrypted image as input.
2. Methodology: 2.1 Dataset Preparation:
• Used the MNIST dataset of handwritten digits.
• Selected a subset of 60,000 images from the MNIST dataset.
• Encrypted 30,000 images using Triple DES (TDES) and another 30,000 using Advanced
Encryption Standard (AES).
• Created a new dataset, EncryptedMNIST_AES_TDES, containing the encrypted images
and their corresponding labels (0 for AES, 1 for TDES).
2.2 Model Architecture:
• Utilized a Convolutional Neural Network (CNN) based on the MobileNetV2
architecture.
• Used transfer learning by employing a pre-trained MobileNetV2 model (trained on
ImageNet) as the base.
• Added custom layers on top of the base model for binary classification.
• Preprocessed images to fit MobileNetV2 input requirements (resizing to 96x96 and
converting to RGB).
2.3 Training Process:
• Split the dataset into training (80%) and testing (20%) sets.
• Implemented a data generator to feed batches of data during training.
• Trained the model for 10 epochs using Adam optimizer and binary cross-entropy loss.
• Used a batch size of 64.
2.4 Evaluation:
• Evaluated the model on the test set.
• Plotted training and validation loss and accuracy curves.
• Tested the model on a sample of 10 random images from the test set, visualizing the
predictions.
3. Results:
• Training and validation loss and accuracy curves were plotted to visualize the learning
process.
• The model demonstrated the ability to distinguish between AES and TDES encrypted
images with reasonable accuracy.
4. Conclusion: The study successfully developed a machine learning model capable of
identifying the encryption algorithm (AES or TDES) used on MNIST images. This
demonstrates the potential of using deep learning techniques in cryptanalysis and
security applications. The use of transfer learning with MobileNetV2 proved effective in
handling the complex patterns in encrypted images..
5.Limitations:
• The study is limited to only two encryption algorithms (AES and TDES).
• The dataset used (MNIST) consists of simple grayscale images, which may not fully
represent real-world scenarios.
• The security implications of this approach need further investigation
![download](https://github.com/user-attachments/assets/caa73f79-85fe-4b8d-9c37-fe45a7cbaf01)

![download (1)](https://github.com/user-attachments/assets/04a0b8d5-ef92-41df-bec4-77d1a43475ca)
![Screenshot 2024-07-31 210357](https://github.com/user-attachments/assets/97b4578b-80f0-49c1-95b8-43c68cadbff0)
