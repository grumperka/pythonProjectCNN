import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
x_train, x_test = x_train.astype('float32'),  x_test.astype('float32')
print(f'Zbiór uczący: {x_train.shape}, zbiór walidacyjny: {x_test.shape}')

img_shape = x_train.shape[1:]
print(img_shape)

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

plt.figure(figsize=(14, 10))
for i in range(40):
    plt.subplot(5, 8, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
plt.show()


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
  #tf.keras.layers.Dropout(0.25),

  tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
  tf.keras.layers.Dropout(0.5),

  tf.keras.layers.Flatten(input_shape=(4, 4, 64)),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
  #tf.keras.layers.Dropout(0.25)
])

model.summary()

predictions = model(x_train[:1]).numpy()
tf.nn.softmax(predictions).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


history = model.fit(x_train, y_train, epochs=20, verbose=1, batch_size=256, validation_split=0.2)


def draw_curves(history, key1='accuracy', ylim1=(0.8, 1.00),
                key2='loss', ylim2=(0.0, 1.0)):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history[key1], "r--")
    plt.plot(history.history['val_' + key1], "g--")
    plt.ylabel('Dokladnosc')
    plt.xlabel('Epoki')
    plt.ylim(ylim1)
    plt.legend(['treningowa', 'testowa'], loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(history.history[key2], "r--")
    plt.plot(history.history['val_' + key2], "g--")
    plt.ylabel('Straty')
    plt.xlabel('Epoki')
    plt.ylim(ylim2)
    plt.legend(['treningowa', 'testowa'], loc='best')

    plt.show()


draw_curves(history, key1='accuracy', ylim1=(0.7, 1),
            key2='loss', ylim2=(0.0, 0.8))

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])

