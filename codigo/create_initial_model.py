"""
Script para criar o modelo inicial plant_model.h5 com MobileNetV2 pré-treinado.
Execute este script uma vez para gerar o modelo necessário para a aplicação funcionar.
"""
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from pathlib import Path

def create_initial_model():
    """Create a MobileNetV2-based plant classifier model."""
    num_classes = 15  # Number of plant disease classes
    
    print("📦 Loading MobileNetV2 base model...")
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    print("🔨 Building classifier head...")
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    print("⚙️  Compiling model...")
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Save the model
    model_path = Path(__file__).parent / "plant_model.h5"
    print(f"💾 Saving model to {model_path}...")
    model.save(str(model_path))
    
    print("✅ Model created successfully!")
    print(f"   Location: {model_path}")
    print(f"   Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    return model

if __name__ == "__main__":
    print("🌱 Creating Plant Vision Initial Model")
    print("=" * 50)
    create_initial_model()
    print("=" * 50)
    print("✨ Your model is ready! The app should now work.")
