# predict.py
import tensorflow as tf
from tensorflow.keras import layers, models
import json
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_model(dataset_path):
    # Create data generator with augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,  # Use 20% data for validation
        fill_mode='nearest'
    )
    
    # Process the nested dataset structure
    class_mapping = {}
    class_dirs = []
    
    # First, locate all the leaf condition subfolders
    for plant_type in os.listdir(dataset_path):
        plant_dir = os.path.join(dataset_path, plant_type)
        if os.path.isdir(plant_dir):
            for condition_folder in os.listdir(plant_dir):
                condition_dir = os.path.join(plant_dir, condition_folder)
                if os.path.isdir(condition_dir):
                    # Add this directory to our list of class directories
                    class_dirs.append(condition_dir)
                    
                    # Parse the folder name to extract plant type and condition
                    condition_parts = condition_folder.split('_')
                    if len(condition_parts) > 1:
                        plant = condition_parts[0]
                        condition = ' '.join(condition_parts[1:])
                        class_mapping[condition_dir] = f"{plant} - {condition}"
                    else:
                        class_mapping[condition_dir] = condition_folder
    
    # Create a temporary flat directory structure for TensorFlow to process
    temp_dataset_path = os.path.join(os.path.dirname(dataset_path), "temp_dataset")
    os.makedirs(temp_dataset_path, exist_ok=True)
    
    # Create symbolic links or copy files to the temporary structure
    class_indices = {}
    for i, class_dir in enumerate(class_dirs):
        class_name = os.path.basename(class_dir)
        temp_class_dir = os.path.join(temp_dataset_path, class_name)
        os.makedirs(temp_class_dir, exist_ok=True)
        
        # Copy or link files
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(class_dir, img_file)
                dst_path = os.path.join(temp_class_dir, img_file)
                
                # Either create a symbolic link (more efficient) or copy the file
                try:
                    os.symlink(os.path.abspath(src_path), dst_path)
                except (OSError, AttributeError):
                    # If symlinks aren't supported, copy the file
                    import shutil
                    shutil.copy2(src_path, dst_path)
        
        # Store class index mapping
        class_indices[class_name] = i
    
    # Now use the flattened dataset structure for training
    train_dataset = train_datagen.flow_from_directory(
        temp_dataset_path,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_dataset = train_datagen.flow_from_directory(
        temp_dataset_path,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Get actual class names from the generator
    class_indices = train_dataset.class_indices
    class_names = list(class_indices.keys())
    num_classes = len(class_names)
    
    # Create formatted class names for display
    formatted_class_names = []
    for class_name in class_names:
        parts = class_name.split('_')
        if len(parts) >= 2:
            plant_type = parts[0]
            condition = ' '.join(parts[1:])
            formatted_class_names.append(f"{plant_type} - {condition}")
        else:
            formatted_class_names.append(class_name)
    
    # Ensure the 'ml_model' directory exists
    save_dir = "ml_model"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save class names to JSON file
    class_names_path = os.path.join(save_dir, "class_names.json")
    with open(class_names_path, "w") as f:
        json.dump(formatted_class_names, f)
    
    # Also save the original class names for prediction mapping
    with open(os.path.join(save_dir, "original_class_names.json"), "w") as f:
        json.dump(class_names, f)

    # Define CNN Model
    model = models.Sequential([
        layers.InputLayer(input_shape=(256, 256, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.5),  # Add dropout to reduce overfitting
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile & Train Model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train the model with validation
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=20,
        callbacks=[early_stopping]
    )

    # Save the model
    save_dir = "ml_model"
    model_path = os.path.join(save_dir, "plant_species_model.h5")
    model.save(model_path)
    
    # Clean up temporary dataset directory
    import shutil
    shutil.rmtree(temp_dataset_path, ignore_errors=True)
    
    return model, formatted_class_names

def predict_image(image_path, model_path=None, class_names_path=None):
    """
    Predict plant disease from an image
    """
    # Default paths
    if model_path is None:
        model_path = "ml_model/plant_species_model.h5"
    if class_names_path is None:
        class_names_path = "ml_model/class_names.json"
    
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Load class names
    with open(class_names_path, "r") as f:
        class_names = json.load(f)
    
    # Load and preprocess the image
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
        
        # Apply the same preprocessing as during training
        img_array = img_array / 255.0
        
        # Make prediction
        predictions = model.predict(img_array)
        
        # Get the predicted class index and top 3 predictions
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index] * 100)
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = [
            {
                "class": class_names[idx],
                "confidence": float(predictions[0][idx] * 100)
            }
            for idx in top_indices
        ]
        
        # Get the predicted class name
        predicted_class = class_names[predicted_class_index]
        
        return {
            "prediction": predicted_class,
            "confidence": round(confidence, 2),
            "top_predictions": top_predictions,
            "status": "success"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    dataset_path = "E:\PLANT-DISEASE-DETECTION\dataset"  # Update this path to your dataset location
    model, class_names = train_model(dataset_path)
    print("✅ Model training completed & saved in 'ml_model' folder.")
    print(f"Available classes: {class_names}")