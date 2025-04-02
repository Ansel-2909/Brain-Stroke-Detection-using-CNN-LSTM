import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, TimeDistributed, Flatten, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, f1_score
import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------------------
# Constants and Paths
# ---------------------------
IMAGE_SIZE = (128, 128)
TIME_STEPS = 5
BATCH_SIZE = 32
EPOCHS = 50   # You may increase this if needed
LEARNING_RATE = 0.001
BASE_FOLDER = r"C:\Stroke Detection\dataset"

# ---------------------------
# Data Loading and Preparation
# ---------------------------
def load_images(base_folder):
    images, labels = [], []
    for folder_name in ['Normal', 'Stroke']:
        folder_path = os.path.join(base_folder, folder_name)
        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder_path}' does not exist")
            continue
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.png'):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, IMAGE_SIZE)
                    images.append(img)
                    labels.append(folder_name)
                else:
                    print(f"Warning: Could not load image '{filename}'")
    
    return np.array(images), labels

def prepare_sequences(images, labels):
    sequences, sequence_labels = [], []
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Create sequences of TIME_STEPS consecutive images.
    for i in range(len(images) - TIME_STEPS + 1):
        seq = images[i:i + TIME_STEPS]
        sequences.append(seq)
        # The label for the sequence is taken as the label of the last image.
        sequence_labels.append(encoded_labels[i + TIME_STEPS - 1])
    
    sequences = np.array(sequences)
    sequence_labels = np.array(sequence_labels)
    return sequences, sequence_labels, label_encoder

# ---------------------------
# Plot Functions
# ---------------------------
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, label_encoder):
    conf_mat = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(label_encoder.classes_))
    plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
    plt.yticks(tick_marks, label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_test, y_pred_prob, class_index=1):
    # For binary classification, we can plot ROC for the positive class.
    fpr, tpr, _ = roc_curve(y_test[:, class_index], y_pred_prob[:, class_index])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# ---------------------------
# Main Data Pipeline
# ---------------------------
# Load images and corresponding labels
images, labels = load_images(BASE_FOLDER)

# Prepare sequences and encode labels
sequences, sequence_labels, label_encoder = prepare_sequences(images, labels)

# Normalize images and add channel dimension
sequences = sequences.astype('float32') / 255.0
# Reshape to (num_samples, TIME_STEPS, height, width, 1)
sequences = sequences.reshape(sequences.shape[0], TIME_STEPS, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sequences, sequence_labels, test_size=0.2, random_state=42)

# One-hot encode labels
num_classes = len(label_encoder.classes_)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test  = tf.keras.utils.to_categorical(y_test, num_classes)

print("Sequences shape:", sequences.shape)

# ---------------------------
# (Optional) Data Augmentation
# ---------------------------
def augment_images(images, labels):
    augmented_images, augmented_labels = [], []
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    
    images = images.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)  # Reshape to 4D for augmentation
    for i in range(len(images)):
        img = images[i]
        label = labels[i]
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        for batch in datagen.flow(img, batch_size=1):
            augmented_images.append(batch[0])
            augmented_labels.append(label)
            if len(augmented_images) >= len(images):
                break

    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)
    return augmented_images, augmented_labels

# Uncomment below if you wish to use augmented data
augmented_images, augmented_labels = augment_images(images, labels)
sequences_aug, sequence_labels_aug, _ = prepare_sequences(augmented_images, augmented_labels)
sequences_aug = sequences_aug.astype('float32') / 255.0
sequences_aug = sequences_aug.reshape(sequences_aug.shape[0], TIME_STEPS, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
X_train_aug, X_test_aug, y_train_aug, y_test_aug = train_test_split(sequences_aug, sequence_labels_aug, test_size=0.2, random_state=42)
y_train_aug = tf.keras.utils.to_categorical(y_train_aug, num_classes)
y_test_aug = tf.keras.utils.to_categorical(y_test_aug, num_classes)
X_train, X_test, y_train, y_test = X_train_aug, X_test_aug, y_train_aug, y_test_aug

# ---------------------------
# Define the Improved CNN-LSTM Model
# ---------------------------
model = Sequential([
    # Convolutional Block 1
    TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'),
                    input_shape=(TIME_STEPS, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling2D((2, 2))),
    
    # Convolutional Block 2
    TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling2D((2, 2))),
    
    # Convolutional Block 3
    TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling2D((2, 2))),
    
    TimeDistributed(Flatten()),
    LSTM(128, return_sequences=True),
    Dropout(0.5),
    LSTM(64),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ---------------------------
# Train the Model
# ---------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[ 
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
    ]
)

# ---------------------------
# Evaluate the Model
# ---------------------------
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Generate predictions
y_pred_prob = model.predict(X_test)
y_pred_class = np.argmax(y_pred_prob, axis=1)
y_true_class = np.argmax(y_test, axis=1)

# Print classification report (includes precision, recall, F1-score)
class_report = classification_report(y_true_class, y_pred_class, target_names=label_encoder.classes_)
print("Classification Report:")
print(class_report)

# Calculate F1 Score (macro average)
f1 = f1_score(y_true_class, y_pred_class, average='macro')
print(f"F1 Score (macro): {f1:.4f}")

# Plot Confusion Matrix
plot_confusion_matrix(y_true_class, y_pred_class, label_encoder)

# Plot ROC curve for the positive class (assuming index 1 corresponds to 'Stroke')
plot_roc_curve(y_test, y_pred_prob, class_index=1)

# Plot Training History (loss & accuracy)
plot_training_history(history)

# Save the trained model
model.save('stroke_detection_model.keras')
print("Model saved as 'stroke_detection_model.keras'")
