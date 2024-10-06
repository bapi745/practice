# Import necessary libraries
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

# Load Fashion MNIST dataset
(train_pictures, train_labels), (test_pictures, test_labels) = fashion_mnist.load_data()

# Preprocess the data (reshape and normalize)
train_pictures = train_pictures[:12000].reshape((12000, 28 * 28)).astype('float32') / 255
test_pictures = test_pictures[:6000].reshape((6000, 28 * 28)).astype('float32') / 255

# One-hot encode the labels
train_labels = to_categorical(train_labels[:12000])
test_labels = to_categorical(test_labels[:6000])

# Build the neural network model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_shape=(28 * 28,)))  # Input layer with 64 neurons
model.add(Dense(units=64, activation='relu'))                          # Hidden layer with 64 neurons
model.add(Dense(units=10, activation='softmax'))                       # Output layer with 10 neurons (classes)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_pictures, train_labels,
                    epochs=12,
                    batch_size=128,
                    validation_data=(test_pictures, test_labels))

# Plot training and validation loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training', 'Validation'], loc='upper left')

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Training', 'Validation'], loc='upper left')

# Show plots
plt.tight_layout()
plt.show()
