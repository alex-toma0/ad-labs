import numpy as np
import keras
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class Autoencoder(keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = keras.Sequential([
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(5, activation='relu'),
            keras.layers.Dense(3, activation='relu')
        ])
        self.decoder = keras.Sequential([
            keras.layers.Dense(5, activation='relu'),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(9, activation='sigmoid')
        ])
    def call(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded

data = loadmat("shuttle.mat")
X = data['X']
y = data['y'].ravel()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
scaler = MinMaxScaler(feature_range=(0,1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

autoencoder = Autoencoder()
autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(
    X_train, X_train,
    epochs=100,
    batch_size=1024,
    validation_data=(X_test, X_test)
                           )
plt.figure(figsize=(10, 6))
plt.grid(True)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

contamination = np.mean(y_train)

reconstructed_samples = autoencoder.predict(X_train)
reconstruction_errors = [mean_squared_error(x, X_train[idx])  for idx, x in enumerate(reconstructed_samples)]

threshold = np.quantile(reconstruction_errors, 1 - contamination)
threshold = np.float64(threshold)
classified_errors = (reconstruction_errors > threshold).astype(int)

balanced_accuracy = balanced_accuracy_score(y_train, classified_errors)
print(f'Balanced accuracy of training set: {balanced_accuracy} ')