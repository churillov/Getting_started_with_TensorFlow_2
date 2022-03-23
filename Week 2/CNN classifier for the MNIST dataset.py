import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def scale_mnist_data(train_images, test_images):
    """
    Эта функция принимает тренировочные и тестовые изображения, и масштабирует их.
    так, чтобы они имели минимальное и максимальное значения, равные 0 и 1 соответственно.
    """
    train_images = train_images / 255
    test_images = test_images / 255

    return train_images, test_images


def get_model(input_shape):
    """
    Функция должна построить последовательную модель
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, (3, 3), padding='SAME', activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    return model


def compile_model(model):
    """
    Функция принимает модель, возвращенную вашей функцией get_model, и компилирует ее с помощью оптимизатора.
    """

    opt = tf.keras.optimizers.Adam()

    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=["accuracy"])


def train_model(model, scaled_train_images, train_labels):
    """
    Эта функция должна обучать модель в течение 5 эпох на scaled_train_images и train_labels.
    Функция должна возвращать историю обучения, возвращаемую model.fit.
    """

    history = model.fit(scaled_train_images, train_labels, epochs=5, batch_size=256)

    return history


def evaluate_model(model, scaled_test_images, test_labels):
    """
    Эта функция должна оценивать модель на scaled_test_images и test_labels.
    Функция должна возвращать кортеж (test_loss, test_accuracy).
    """

    return model.evaluate(scaled_test_images, test_labels, verbose=5)


mnist_data = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_data.load_data()

scaled_train_images, scaled_test_images = scale_mnist_data(train_images, test_images)
scaled_train_images = scaled_train_images[..., np.newaxis]
scaled_test_images = scaled_test_images[..., np.newaxis]

model = get_model(scaled_train_images[0].shape)
model.summary()

compile_model(model)

history = train_model(model, scaled_train_images, train_labels)

frame = pd.DataFrame(history.history)
frame.head()

acc_plot = frame.plot(y="accuracy", title="Accuracy vs Epochs", legend=False)
acc_plot.set(xlabel="Epochs", ylabel="Accuracy")

acc_plot = frame.plot(y="loss", title="Loss vs Epochs", legend=False)
acc_plot.set(xlabel="Epochs", ylabel="Loss")

test_loss, test_accuracy = evaluate_model(model, scaled_test_images, test_labels)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")

num_test_images = scaled_test_images.shape[0]

random_inx = np.random.choice(num_test_images, 4)
random_test_images = scaled_test_images[random_inx, ...]
random_test_labels = test_labels[random_inx, ...]

predictions = model.predict(random_test_images)

fig, axes = plt.subplots(4, 2, figsize=(16, 12))
fig.subplots_adjust(hspace=0.4, wspace=-0.2)

for i, (prediction, image, label) in enumerate(zip(predictions, random_test_images, random_test_labels)):
    axes[i, 0].imshow(np.squeeze(image))
    axes[i, 0].get_xaxis().set_visible(False)
    axes[i, 0].get_yaxis().set_visible(False)
    axes[i, 0].text(10., -1.5, f'Digit {label}')
    axes[i, 1].bar(np.arange(len(prediction)), prediction)
    axes[i, 1].set_xticks(np.arange(len(prediction)))
    axes[i, 1].set_title(f"Categorical distribution. Model prediction: {np.argmax(prediction)}")

plt.show()
