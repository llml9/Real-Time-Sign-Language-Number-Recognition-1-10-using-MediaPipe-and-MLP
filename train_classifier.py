import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Encode labels into integers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split data into train/test sets
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.3, shuffle=True, stratify=labels
)

# -------- Build MLP model with Gaussian Noise and Dropout --------
model = tf.keras.Sequential([
    tf.keras.layers.GaussianNoise(0.1, input_shape=(x_train.shape[1],)),  # add Gaussian noise to inputs
    tf.keras.layers.Dropout(0.5),  # randomly drops 40% neurons
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.4),  # randomly drops 30% neurons
    tf.keras.layers.Dense(len(np.unique(labels)), activation='softmax')  # output layer
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model (reduced epochs to avoid overfitting)
history = model.fit(x_train, y_train, epochs=20, batch_size=32,
                    validation_data=(x_test, y_test))

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy (MLP with Gaussian Noise + Dropout)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()

# Evaluate the model
y_predict = np.argmax(model.predict(x_test), axis=-1)
score = accuracy_score(y_test, y_predict)
print('Model accuracy: {:.2f}% of the samples were classified correctly!'.format(score * 100))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_predict)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix (MLP)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Save the model
model.save('model_mlp.h5')

