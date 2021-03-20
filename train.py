"""
Michael Patel
March 2021

Project description:
    Build an image classifier of cats and dogs using transfer learning: feature extraction

"""
################################################################################
# Imports
import os
import matplotlib.pyplot as plt
import tensorflow as tf


################################################################################
# Main
if __name__ == "__main__":
    # ----- ETL ----- #

    _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
    PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

    train_dir = os.path.join(PATH, 'train')
    validation_dir = os.path.join(PATH, 'validation')

    BATCH_SIZE = 32
    IMAGE_WIDTH = 160
    IMAGE_HEIGHT = 160
    IMAGE_CHANNELS = 3

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=train_dir,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=(IMAGE_WIDTH, IMAGE_HEIGHT)
    )

    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=validation_dir,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=(IMAGE_WIDTH, IMAGE_HEIGHT)
    )

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    # ----- MODEL ----- #
    # transfer learning: feature extraction
    data_augment_layer = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
    ])

    # base model: MobileNetV2
    preprocess_input_layer = tf.keras.applications.mobilenet_v2.preprocess_input
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
        weights="imagenet",
        include_top=False
    )
    base.trainable = False

    # add classification head
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    dropout_layer = tf.keras.layers.Dropout(0.2)
    output_layer = tf.keras.layers.Dense(units=1)

    # build model
    inputs = tf.keras.Input(
        shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    )
    x = inputs
    x = data_augment_layer(x)
    x = preprocess_input_layer(x)
    x = base(x, training=False)
    x = global_average_layer(x)
    x = dropout_layer(x)
    x = output_layer(x)
    outputs = x

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=["accuracy"]
    )

    model.summary()

    # ----- TRAIN ----- #
    epochs = 10
    history = model.fit(
        x=train_dataset,
        epochs=epochs,
        validation_data=validation_dataset
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    #plt.show()

    # save model
    SAVE_DIR = os.path.join(os.getcwd(), "saved_model")
    model.save(SAVE_DIR)

    # ----- DEPLOY ----- #
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVE_DIR)
    tflite_model = converter.convert()

    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
