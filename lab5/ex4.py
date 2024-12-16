import keras
import tensorflow as tf 
from sklearn.metrics import mean_squared_error, balanced_accuracy_score
import numpy as np
import matplotlib.pyplot as plt

class ConvAutoencoder(keras.Model):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = keras.Sequential([
            keras.layers.Conv2D(8, (3,3), activation='relu', strides=2, padding='same'),
            keras.layers.Conv2D(4, (3,3), activation='relu', strides=2, padding='same')
        ])
        self.decoder = keras.Sequential([
            keras.layers.Conv2DTranspose(4, (3,3), activation='relu', strides=2, padding='same'),
            keras.layers.Conv2DTranspose(8, (3,3), activation='relu', strides=2, padding='same'),
            keras.layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')

        ])
    def call(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded


(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

y_test_binary = np.zeros_like(y_test)
# Reshape for Conv layers
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)


X_train = X_train / 255.0
X_test = X_test / 255.0


# Add noise
X_train_noise  = X_train + tf.random.normal(shape=X_train.shape) * 0.35
X_train_noise  = tf.clip_by_value(X_train_noise, clip_value_min=0, clip_value_max=1)

X_test_noise  = X_test + tf.random.normal(shape=X_test.shape) * 0.35
X_test_noise = tf.clip_by_value(X_test_noise, clip_value_min=0, clip_value_max=1)
  
conv_autoencoder = ConvAutoencoder()
conv_autoencoder.compile(optimizer='adam', loss='mse')

history = conv_autoencoder.fit(X_train, X_train,
                               epochs=10,
                               batch_size=64,
                               validation_data=(X_test, X_test))

rec_train = conv_autoencoder.predict(X_train)
rec_errors_train = keras.losses.mean_squared_error(rec_train, X_train)
threshold = np.mean(rec_errors_train) + np.std(rec_errors_train)

rec_test = conv_autoencoder.predict(X_test)
rec_errors_test = keras.losses.mean_squared_error(rec_test, X_test)

rec_test_noise = conv_autoencoder.predict(X_test_noise)
rec_errors_test_noise = keras.losses.mean_squared_error(rec_test_noise, X_test_noise)

img_errors_test = tf.reduce_mean(rec_errors_test, axis=(1,2))
img_errors_test_noise = tf.reduce_mean(rec_errors_test_noise, axis=(1,2))

classified_test = img_errors_test < threshold
classified_test_noise = img_errors_test_noise < threshold

ba_test = balanced_accuracy_score(y_test, classified_test._numpy().astype(int))
ba_test_noise = balanced_accuracy_score(y_test, classified_test_noise._numpy().astype(int))

print(f"Original test images accuracy: {ba_test}")
print(f"Test images with added noise accuracy: {ba_test_noise}")

fig, axs = plt.subplots(4, 5, figsize=(15, 8))

for i in range(5):
   
    axs[0, i].imshow(tf.squeeze(X_test[i]), cmap='gray')
    axs[0, i].axis('off')
    axs[0, i].set_title("original")

    axs[1, i].imshow(tf.squeeze(X_test_noise[i]), cmap='gray')
    axs[1, i].axis('off')
    axs[1, i].set_title("noisy")

    axs[2, i].imshow(tf.squeeze(rec_test[i]), cmap='gray')
    axs[2, i].axis('off')
    axs[2, i].set_title("reconstructed")

    axs[3, i].imshow(tf.squeeze(rec_test_noise[i]), cmap='gray')
    axs[3, i].axis('off')
    axs[3, i].set_title("reconstructed (noisy)")

plt.show()