import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_plant_classifier(
    num_classes: int,
    fine_tune_from: int = 100
):

    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model

def enable_fine_tuning(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    from_layer: int = 100
):
    base_model.trainable = True

    for layer in base_model.layers[:from_layer]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

def diagnose_plant(
    model: tf.keras.Model,
    img_tensor: np.ndarray,
    class_names: list,
    confidence_threshold: float = 0.6
) -> dict:

    probs = model.predict(img_tensor, verbose=0)[0]

    top3_idx = np.argsort(probs)[::-1][:3]

    predicted_class = class_names[top3_idx[0]]
    confidence = float(probs[top3_idx[0]])

    return {
        "diagnosis": predicted_class,
        "confidence": confidence,
        "is_reliable": confidence >= confidence_threshold,
        "top3": [
            {"class": class_names[i], "prob": float(probs[i])}
            for i in top3_idx
        ]
    }