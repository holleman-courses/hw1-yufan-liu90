#!/usr/bin/env python

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import Input, layers, Sequential

# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image

print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")


##

def build_model1():
    model = keras.Sequential(
        [
            layers.Flatten(input_shape=(32, 32, 3)),
            layers.Dense(128, activation=layers.LeakyReLU(alpha=0.3)),
            layers.Dense(128, activation=layers.LeakyReLU(alpha=0.3)),
            layers.Dense(128, activation=layers.LeakyReLU(alpha=0.3)),
            layers.Dense(10),  # logits
        ],
        name="model1",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def build_model2():
    model = keras.Sequential(
        [
            layers.Conv2D(
                32, (3, 3), strides=2, padding="same",
                activation="relu", input_shape=(32, 32, 3)
            ),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), strides=2, padding="same", activation="relu"),
            layers.BatchNormalization(),

            layers.Conv2D(128, (3, 3), strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), strides=1, padding="same"),
            layers.BatchNormalization(),

            layers.Flatten(),
            layers.Dense(10),  # logits
        ],
        name="model2",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def build_model3():
    model = keras.Sequential(
        [
            layers.SeparableConv2D(
                32, (3, 3), strides=2, padding="same",
                input_shape=(32, 32, 3)
            ),
            layers.BatchNormalization(),
            layers.SeparableConv2D(64, (3, 3), strides=2, padding="same"),
            layers.BatchNormalization(),

            layers.SeparableConv2D(128, (3, 3), strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.SeparableConv2D(128, (3, 3), strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.SeparableConv2D(128, (3, 3), strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.SeparableConv2D(128, (3, 3), strides=1, padding="same"),
            layers.BatchNormalization(),

            layers.Flatten(),
            layers.Dense(10),  # logits
        ],
        name="model3",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def build_model50k():
    model = inputs = keras.Input(shape=(32, 32, 3))

    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv2D(64, 3, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv2D(96, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv2D(128, 3, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Dropout(0.2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10)(x)  # logits

    model = keras.Model(inputs, outputs, name="model50k")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import
if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="best_model.h5")
    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    ########################################
    # Add code here to Load the CIFAR10 data set
    (train_images, train_labels), (test_images, test_labels) = \
        tf.keras.datasets.cifar10.load_data()

    train_labels = train_labels.squeeze()
    test_labels = test_labels.squeeze()

    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    # Validation split (10%)
    val_size = int(0.1 * train_images.shape[0])
    val_images = train_images[:val_size]
    val_labels = train_labels[:val_size]
    tr_images = train_images[val_size:]
    tr_labels = train_labels[val_size:]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((tr_images, tr_labels))
        .shuffle(20000, seed=args.seed)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    ########################################
    # Build and train model 1
    # model1 = build_model1()
    # compile and train model 1.

    # Build, compile, and train model 2 (DS Convolutions)
    # model2 = build_model2()

    # Repeat for model 3 and your best sub-50k params model
    # model3 = build_model3()
    print("\nTraining sub-50k model...")
    best = build_model50k()
    print("model50k params:", best.count_params())

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=3,
            min_lr=1e-5, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            args.save_path, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
    ]

    best.fit(train_ds, validation_data=val_ds,
             epochs=args.epochs, callbacks=callbacks, verbose=2)

    loss, acc = best.evaluate(test_ds, verbose=0)
    print(f"Final test accuracy: {acc:.4f}")

    if not os.path.exists(args.save_path):
        best.save(args.save_path)
        print("Saved:", args.save_path)
