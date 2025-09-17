import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import cv2
import numpy as np

# --- Bagian 1: Preprocessing Data ---
# Direktori dataset
train_dir = "dataset/seg_train/seg_train"
val_dir = "dataset/seg_test/seg_test"

# Augmentasi data
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# --- Bagian 2: Training Model CNN ---
# Definisi model CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax')
])

# Kompilasi model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Simpan model
model.save('cnn_model.h5')

# --- Bagian 3: Integrasi dengan OpenCV untuk Prediksi Real-time ---
# Load model yang telah dilatih
model = load_model('cnn_model.h5')

# Load label kelas
class_labels = list(train_generator.class_indices.keys())

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Mode Night Vision
    night_vision = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    night_vision = cv2.applyColorMap(night_vision, cv2.COLORMAP_JET)
    
    # Preprocessing gambar
    img = cv2.resize(frame, (150, 150))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Prediksi kelas
    pred = model.predict(img)
    label = class_labels[np.argmax(pred)]
    
    # Tampilkan hasil
    cv2.putText(frame, f'Class: {label}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    cv2.imshow('Night Vision', night_vision)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()8