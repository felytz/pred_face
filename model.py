import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import albumentations as A
import gc

# --- Constants ---
IMG_SIZE = (224, 224)  # ResNet50 default input size
BATCH_SIZE = 8  # Reduced for CPU-only training
EPOCHS = 2 #20
NUM_CLASSES = 11 

# --- Data Augmentation with Albumentations ---
augmenter = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=20, p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.1),
])

def augment_image(image):
    augmented = augmenter(image=image)
    return augmented['image']

# --- Load & Augment Data ---
def load_and_split_data():
    X, y = [], []
    root_dir = "256x256"
    class_names = sorted(os.listdir(root_dir))
    
    for label, name in enumerate(class_names):
        class_dir = os.path.join(root_dir, name)
        frames = sorted([f for f in os.listdir(class_dir) if f.endswith(".png")])
        
        for frame in frames:
            img_path = os.path.join(class_dir, frame)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            X.append(img)
            y.append(label)
    
    X = np.array(X) / 255.0
    y = np.array(y)
    
    # Assign groups (every 5 frames = 1 group)
    groups = [i // 5 for i in range(len(X))]
    
    # Split with GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# Load data
X_train, X_test, y_train, y_test = load_and_split_data()

# --- Build ResNet50 Model ---
def build_resnet50_model(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)
    base = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
    
    # Freeze initial layers
    for layer in base.layers[:100]:
        layer.trainable = False
    
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Create model
print("\n--- Training ResNet50 on CPU ---")
model = build_resnet50_model()

# Add callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# --- Plot and Save Results ---
output_dir = "training_results"
os.makedirs(output_dir, exist_ok=True)

# 1. Accuracy/Loss Plot
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'accuracy_loss.png'))
plt.close()

# 2. Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()

# 3. Training Samples per Class
plt.figure(figsize=(10, 6))
sns.countplot(x=y_train)
plt.title('Training Data Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
plt.close()

# 4. Save classification report as text file
with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
    f.write("Classification Report:\n")
    f.write(report)

print(f"\nAll results saved to directory: {output_dir}")
print(f"Contents:")
print(f"- accuracy_loss.png (training metrics)")
print(f"- confusion_matrix.png")
print(f"- class_distribution.png")
print(f"- classification_report.txt")

# Save model
try:
    model.save("resnet50_cpu_model.keras")
    print("Model saved successfully as .keras")
except Exception as e:
    print(f"Error saving model: {e}")
try:
    model.export("model")
    print("Model saved successfully as .keras")
except Exception as e:
    print(f"Error saving model as folder: {e}")

# Clear memory
del model
gc.collect()
tf.keras.backend.clear_session()