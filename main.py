import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

mnist = tf.keras.datasets.mnist

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
) #manipulowanie obrazem, obrót, powiększenie, zmniejszenie

#pobieranie danych
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#formatowanie rozmiaru obrazów do rozmiaru 28x28x1
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

#normalizowanie pixeli do postaci 0 - 1
x_train, x_test = x_train.astype('float32') / 255.0,  x_test.astype('float32') / 255.0

print(f'Zbiór uczący: {x_train.shape}, zbiór walidacyjny: {x_test.shape}')
#zbiór uczący 60 000
#zbiór walidacyjny 10 000

img_shape = x_train.shape[1:]
print(f'Wymiary obrazu: {img_shape}')

#labele do obrazków
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

#funkcja do wyświetlenia 40 obrazów ze zbioru treningowego
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
  tf.keras.layers.MaxPool2D(pool_size=(2, 2)), #zmniejsza rozdzielczość map cech
  tf.keras.layers.Dropout(0.25),

  tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
  tf.keras.layers.Dropout(0.5), #zapobiega uczeniu się na pamięć

  tf.keras.layers.Flatten(input_shape=(4, 4, 64)), #spłaszcza tensor do 1D
  tf.keras.layers.Dense(64, activation='relu'), #max(0,x)
  tf.keras.layers.Dense(10, activation='softmax') #sprawdza, czy prawdopodobieństwa sumują się do 1
])

model.summary()

print('################################')

predictions = model(x_train[:1]).numpy() #Dla każdego przykładu model zwraca 1 wektor wyników „ logits ” lub „ log-odds ”
tf.nn.softmax(predictions).numpy() #konwersja logitów na prawdopowobieńswo
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) #zwraca stratę skalarną dla każdego przykłady
loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

history = model.fit(datagen.flow(x_train, y_train, batch_size=64), validation_data=(x_test, y_test), epochs=20, verbose=1)

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

print('################################')

model.evaluate(x_test,  y_test, verbose=2) #sprawdzanie wydajności modelu na danych walidacyjnych
print('################################')

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax() #dodawanie warstwy Softmax do zwrócenia prawdopodobieństw
])

probability_model(x_test[:5])

