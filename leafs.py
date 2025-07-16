import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalMaxPooling2D, LeakyReLU, ELU
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve, auc, roc_curve
import os
import random
from PIL import Image
import keras_tuner as kt
import shutil
import logging
import math 

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

base_path = 'leafs'
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 8
EPOCHS = 30 
NUM_FOLDS = 5

early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='min', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='min', min_lr=0.00001)
callbacks = [early_stopping, reduce_lr]

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest' 
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

try:
    train_generator = train_datagen.flow_from_directory(
        os.path.join(base_path, 'Train'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    val_generator = val_datagen.flow_from_directory(
        os.path.join(base_path, 'Validation'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    test_generator = test_datagen.flow_from_directory(
        os.path.join(base_path, 'Test'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
except Exception as e:
    logging.error(f"Error loading data generators: {e}")
    raise

CLASS_NAMES = list(train_generator.class_indices.keys())
NUM_CLASSES = len(CLASS_NAMES)
logging.info(f"Number of training samples: {train_generator.samples}")
logging.info(f"Number of validation samples: {val_generator.samples}")
logging.info(f"Number of test samples: {test_generator.samples}")
logging.info(f"Class names: {CLASS_NAMES}")

def visualize_samples(directory, class_names, num_samples=3):
    plt.figure(figsize=(15, 5))
    for i, class_name in enumerate(class_names):
        class_path = os.path.join(directory, class_name)
        image_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        sample_images = random.sample(image_files, min(num_samples, len(image_files)))
        for j, img_path in enumerate(sample_images):
            plt.subplot(len(class_names), num_samples, i * num_samples + j + 1)
            img = Image.open(img_path)
            plt.imshow(img)
            plt.title(class_name)
            plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_class_distribution(generator, title):
    class_labels = generator.classes
    class_counts = pd.Series(class_labels).value_counts().sort_index()
    class_names = list(generator.class_indices.keys())
    plt.figure(figsize=(8, 6))
    plt.bar(class_names, class_counts.values, color=['blue', 'orange', 'green'])
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.show()

def plot_augmented_samples(train_datagen_instance, sample_image, num_augmented_images=5):
    augmented_images = [train_datagen_instance.random_transform(sample_image) for _ in range(num_augmented_images)]

    plt.figure(figsize=(15, 5))
    plt.subplot(1, num_augmented_images + 1, 1)
    plt.imshow(sample_image)
    plt.title('Original Image')
    plt.axis('off')
    for i, aug_img in enumerate(augmented_images):
        plt.subplot(1, num_augmented_images + 1, i + 2)
        aug_img = np.clip(aug_img, 0.0, 1.0) 
        aug_img = (aug_img * 255).astype(np.uint8) 
        plt.imshow(aug_img)
        plt.title(f'Augmented {i+1}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

logging.info("--- Visualizing Sample Images ---")
visualize_samples(os.path.join(base_path, 'Train'), CLASS_NAMES)
visualize_samples(os.path.join(base_path, 'Test'), CLASS_NAMES)

logging.info("--- Plotting Class Distributions ---")
plot_class_distribution(train_generator, 'Training Set Class Distribution')
plot_class_distribution(val_generator, 'Validation Set Class Distribution')
plot_class_distribution(test_generator, 'Test Set Class Distribution')

logging.info("--- Visualizing Data Augmentation ---")
sample_image, _ = train_generator.__next__()
sample_image = sample_image[0] 
plot_augmented_samples(train_datagen, sample_image)

def create_model(base_model_class, input_shape, num_classes, optimizer_instance):
    model = Sequential()
    model.add(base_model_class(weights='imagenet', include_top=False, input_shape=input_shape))
    
    base_model_layers = model.layers[0].layers
    for layer in base_model_layers[:-20]: 
        layer.trainable = False

    model.add(GlobalMaxPooling2D()) 

    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(256, kernel_regularizer=regularizers.l2(0.0001)))
    model.add(LeakyReLU(negative_slope=0.01)) 
    model.add(Dropout(0.3))

    model.add(Dense(128, kernel_regularizer=regularizers.l2(0.0001)))
    model.add(ELU(alpha=1.0)) 
    model.add(Dropout(0.4)) 

    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.0005)))
    model.add(BatchNormalization()) 
    model.add(Dropout(0.2))

    model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.0005)))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=optimizer_instance, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

logging.info("--- Training Initial MobileNetV2 Model ---")
initial_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
mobilenetv2_initial_model = create_model(MobileNetV2, (IMG_HEIGHT, IMG_WIDTH , 3), NUM_CLASSES, initial_optimizer)

mobilenetv2_initial_model.summary()

initial_history_mobilenetv2 = mobilenetv2_initial_model.fit(
    train_generator,
    steps_per_epoch=math.ceil(train_generator.samples / BATCH_SIZE), 
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=math.ceil(val_generator.samples / BATCH_SIZE), 
    callbacks=callbacks
)

mobilenetv2_initial_loss, mobilenetv2_initial_acc = mobilenetv2_initial_model.evaluate(test_generator, steps=math.ceil(test_generator.samples / BATCH_SIZE)) 

logging.info(f'Initial MobileNetV2 Test Accuracy: {mobilenetv2_initial_acc:.4f}')
logging.info(f'Initial MobileNetV2 Test Loss: {mobilenetv2_initial_loss:.4f}')

def prepare_data_for_kfold(base_path_kfold):
    all_image_paths = []
    all_labels = []
    
    class_names = sorted(os.listdir(os.path.join(base_path_kfold, 'Train')))
    class_indices = {name: i for i, name in enumerate(class_names)}

    for class_name in class_names:
        class_path = os.path.join(base_path_kfold, 'Train', class_name)
        try:
            for img_name in os.listdir(class_path):
                if os.path.isfile(os.path.join(class_path, img_name)):
                    all_image_paths.append(os.path.join(class_path, img_name))
                    all_labels.append(class_indices[class_name])
        except Exception as e:
            logging.error(f"Error reading class {class_name} from Train directory: {e}")
            raise

    val_class_names = sorted(os.listdir(os.path.join(base_path_kfold, 'Validation')))
    for class_name in val_class_names:
        if class_name not in class_indices:
            logging.warning(f"Class '{class_name}' found in Validation but not in Train. Skipping.")
            continue
        class_path = os.path.join(base_path_kfold, 'Validation', class_name)
        try:
            for img_name in os.listdir(class_path):
                if os.path.isfile(os.path.join(class_path, img_name)):
                    all_image_paths.append(os.path.join(class_path, img_name))
                    all_labels.append(class_indices[class_name])
        except Exception as e:
            logging.error(f"Error reading validation class {class_name}: {e}")
            raise

    return np.array(all_image_paths), np.array(all_labels), class_names

try:
    X, y, CLASS_NAMES = prepare_data_for_kfold(base_path)
    NUM_CLASSES = len(CLASS_NAMES) 
    logging.info(f"Total samples for K-Fold: {len(X)}")
except Exception as e:
    logging.error(f"Error preparing K-Fold data: {e}")
    raise

def build_model_for_tuner(hp):
    lr_schedule = CosineDecay(
        initial_learning_rate=hp.Float('learning_rate', 1e-5, 1e-2, sampling='log'),
        decay_steps= math.ceil((len(X) * (NUM_FOLDS - 1) / NUM_FOLDS) / BATCH_SIZE) * EPOCHS 
    )
    optimizer = Adam(learning_rate=lr_schedule)
    model = create_model(MobileNetV2, (IMG_HEIGHT, IMG_WIDTH, 3), NUM_CLASSES, optimizer)
    return model

skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
kfold_histories = [] 
kfold_val_predictions = [] 
kfold_val_true_labels = [] 
kfold_tuner_results = [] 
kfold_val_accuracies = [] 
kfold_val_losses = [] 
fold_models_paths = [] 
best_kfold_val_acc = -float('inf')
best_kfold_model_path = 'best_kfold_model.h5' 

logging.info("\n--- Starting K-Fold Cross-Validation ---")
for fold, (train_indices, val_indices) in enumerate(skf.split(X, y)):
    logging.info(f"\n--- Starting Fold {fold + 1}/{NUM_FOLDS} ---")
    
    train_df = pd.DataFrame({'filename': X[train_indices], 'class': [CLASS_NAMES[y[i]] for i in train_indices]})
    val_df = pd.DataFrame({'filename': X[val_indices], 'class': [CLASS_NAMES[y[i]] for i in val_indices]})
    
    try:
        fold_train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='filename',
            y_col='class',
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True
        )
        fold_val_generator = val_datagen.flow_from_dataframe(
            val_df,
            x_col='filename',
            y_col='class',
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
    except Exception as e:
        logging.error(f"Error creating data generators for fold {fold + 1}: {e}")
        raise

    tuner = kt.Hyperband(
        build_model_for_tuner,
        objective='val_accuracy',
        max_epochs=EPOCHS,
        factor=3,
        directory=f'hyperband_fold_{fold}', 
        project_name='leaf_classification',
        seed=42
    )
    
    tuner.search(
        fold_train_generator,
        validation_data=fold_val_generator,
        epochs=EPOCHS,
        callbacks=[early_stopping, reduce_lr], 
        verbose=0, 
        steps_per_epoch=math.ceil(fold_train_generator.samples / BATCH_SIZE), 
        validation_steps=math.ceil(fold_val_generator.samples / BATCH_SIZE) 
    )
    
    best_model_for_fold = tuner.get_best_models(num_models=1)[0]
    best_hps_for_fold = tuner.get_best_hyperparameters(num_trials=1)[0]
    logging.info(f"Fold {fold + 1} Best hyperparameters: {best_hps_for_fold.values}")
    
    history = best_model_for_fold.fit(
        fold_train_generator,
        epochs=EPOCHS,
        validation_data=fold_val_generator,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
        steps_per_epoch=math.ceil(fold_train_generator.samples / BATCH_SIZE),
        validation_steps=math.ceil(fold_val_generator.samples / BATCH_SIZE) 
    )
    kfold_histories.append(history)
    
    val_loss, val_acc = best_model_for_fold.evaluate(fold_val_generator, verbose=0)
    logging.info(f"Fold {fold + 1} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    kfold_val_accuracies.append(val_acc) 
    kfold_val_losses.append(val_loss) 

    val_predictions = best_model_for_fold.predict(fold_val_generator, verbose=0)
    kfold_val_predictions.append(val_predictions)
    kfold_val_true_labels.append(fold_val_generator.classes)
    
    trial_results_for_fold = []
    for trial in tuner.oracle.get_best_trials(num_trials=10): 
        trial_results_for_fold.append({
            'hyperparameters': trial.hyperparameters.values,
            'val_accuracy': trial.score
        })
    kfold_tuner_results.append(trial_results_for_fold)

    model_path = f'model_fold_{fold + 1}.h5'
    best_model_for_fold.save(model_path)
    fold_models_paths.append(model_path) 

    if val_acc > best_kfold_val_acc:
        best_kfold_val_acc = val_acc
        best_model_for_fold.save(best_kfold_model_path)
        logging.info(f"Fold {fold + 1} model saved as best K-Fold model with validation accuracy: {val_acc:.4f}")

def plot_training_history(history, fold_num, model_type="Model"):
    """Plots training and validation accuracy and loss."""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_type} {f"Fold {fold_num}" if fold_num else ""} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_type} {f"Fold {fold_num}" if fold_num else ""} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_roc_curve(true_labels, predicted_probabilities, class_names, title):
    plt.figure(figsize=(10, 8))
    num_classes = predicted_probabilities.shape[1]
    
    for i in range(num_classes):
        true_labels_binary = (true_labels == i).astype(int)
        fpr_i, tpr_i, _ = roc_curve(true_labels_binary, predicted_probabilities[:, i])
        auc_i = auc(fpr_i, tpr_i)
        plt.plot(fpr_i, tpr_i, label=f'Class {class_names[i]} (AUC = {auc_i:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier') 
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def plot_hyperparameter_results(trial_results, fold_num):
    if not trial_results:
        logging.warning(f"No hyperparameter trial results available for Fold {fold_num}. Skipping plot.")
        return
    
    learning_rates = [trial['hyperparameters']['learning_rate'] for trial in trial_results]
    val_accuracies = [trial['val_accuracy'] for trial in trial_results]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(learning_rates, val_accuracies, c='blue', alpha=0.6, edgecolors='w', s=80)
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Fold {fold_num} Hyperparameter Tuning: Validation Accuracy vs Learning Rate')
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.show()

def plot_comparative_accuracies(initial_model_acc, kfold_accuracies):
    labels = ['Initial MobileNetV2 Test Accuracy'] + [f'Fold {i+1} Val Acc' for i in range(len(kfold_accuracies))]
    accuracies = [initial_model_acc] + kfold_accuracies

    plt.figure(figsize=(12, 7))
    bars = plt.bar(labels, accuracies, color=['skyblue'] + ['lightcoral'] * len(kfold_accuracies))
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, round(yval, 4), ha='center', va='bottom')

    plt.axhline(y=np.mean(kfold_accuracies) if kfold_accuracies else 0, color='red', linestyle='--', label=f'Average K-Fold Val Acc: {np.mean(kfold_accuracies):.4f}' if kfold_accuracies else '')
    
    plt.ylabel('Accuracy')
    plt.title('Comparative Accuracy of Models')
    plt.ylim(min(0.5, min(accuracies) - 0.05 if accuracies else 0), max(1.0, max(accuracies) + 0.05 if accuracies else 0)) 
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()

def plot_comparative_losses(initial_model_loss, kfold_losses):
    labels = ['Initial MobileNetV2 Test Loss'] + [f'Fold {i+1} Val Loss' for i in range(len(kfold_losses))]
    losses = [initial_model_loss] + kfold_losses

    plt.figure(figsize=(12, 7))
    bars = plt.bar(labels, losses, color=['lightgreen'] + ['salmon'] * len(kfold_losses))
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, round(yval, 4), ha='center', va='bottom')

    plt.axhline(y=np.mean(kfold_losses) if kfold_losses else 0, color='blue', linestyle='--', label=f'Average K-Fold Val Loss: {np.mean(kfold_losses):.4f}' if kfold_losses else '')
    
    plt.ylabel('Loss')
    plt.title('Comparative Loss of Models')
    plt.ylim(0, max(losses) + 0.1 if losses else 1.0) 
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()

def ensemble_predict(model_paths, generator):
    predictions = []
    for model_path in model_paths:
        try:
            model = tf.keras.models.load_model(model_path)
            preds = model.predict(generator, verbose=0)
            predictions.append(preds)
        except Exception as e:
            logging.error(f"Error loading model {model_path} for ensembling: {e}")
            continue
    return np.mean(predictions, axis=0) if predictions else None


logging.info("\n--- Generating Plots for Initial MobileNetV2 Model ---")
plot_training_history(initial_history_mobilenetv2, fold_num=None, model_type="Initial MobileNetV2")
initial_preds = mobilenetv2_initial_model.predict(test_generator, verbose=0)
plot_roc_curve(test_generator.classes, initial_preds, CLASS_NAMES, 'Initial MobileNetV2 Test Set ROC Curve')

logging.info("\n--- Generating Plots for Each K-Fold Model ---")
for i, history in enumerate(kfold_histories):
    logging.info(f"\nDisplaying plots for Fold {i + 1}")
    plot_training_history(history, fold_num=i+1, model_type="K-Fold Model")
    plot_roc_curve(kfold_val_true_labels[i], kfold_val_predictions[i], CLASS_NAMES, f'Fold {i + 1} ROC Curve (Validation Set)')
    plot_hyperparameter_results(kfold_tuner_results[i], fold_num=i+1)

logging.info("\n--- Generating Comparative Accuracy and Loss Plots ---")
plot_comparative_accuracies(mobilenetv2_initial_acc, kfold_val_accuracies)
plot_comparative_losses(mobilenetv2_initial_loss, kfold_val_losses) 

logging.info("\n--- Final Ensemble Evaluation on Test Set ---")
try:
    ensemble_preds = ensemble_predict(fold_models_paths, test_generator)
    if ensemble_preds is None:
        raise ValueError("No valid model predictions for ensembling")
    y_pred_classes = np.argmax(ensemble_preds, axis=1)
    y_true_classes = test_generator.classes
    
    print("\nClassification Report (Ensemble):")
    print(classification_report(y_true_classes, y_pred_classes, target_names=CLASS_NAMES))
    print("\nConfusion Matrix (Ensemble):")
    print(confusion_matrix(y_true_classes, y_pred_classes))
    print(f"\nWeighted F1-Score (Ensemble): {f1_score(y_true_classes, y_pred_classes, average='weighted'):.4f}")
    
    plt.figure(figsize=(10, 8))
    for i in range(NUM_CLASSES):
        precision, recall, _ = precision_recall_curve((y_true_classes == i).astype(int), ensemble_preds[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'Class {CLASS_NAMES[i]} (PR AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Ensemble Test Set)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plot_roc_curve(y_true_classes, ensemble_preds, CLASS_NAMES, 'ROC Curve (Ensemble Test Set)')

except Exception as e:
    logging.error(f"Error during test set evaluation: {e}")

logging.info("\n--- K-Fold Validation Summary ---")
for i, acc in enumerate(kfold_val_accuracies):
    logging.info(f"Fold {i+1}: Val Acc = {acc:.4f}, Val Loss = {kfold_val_losses[i]:.4f}")

avg_kfold_val_acc = np.mean(kfold_val_accuracies) if kfold_val_accuracies else 0
avg_kfold_val_loss = np.mean(kfold_val_losses) if kfold_val_losses else 0
logging.info(f"\nAverage K-Fold Validation Accuracy: {avg_kfold_val_acc:.4f}")
logging.info(f"Average K-Fold Validation Loss: {avg_kfold_val_loss:.4f}")

best_overall_model_path = 'best_overall_leaf_model.h5'

if mobilenetv2_initial_acc >= avg_kfold_val_acc: 
    logging.info(f"\nInitial MobileNetV2 model (Test Acc: {mobilenetv2_initial_acc:.4f}) performed better or equal to K-Fold average (Val Acc: {avg_kfold_val_acc:.4f}). Saving initial model.")
    mobilenetv2_initial_model.save(best_overall_model_path)
else:
    logging.info(f"\nK-Fold cross-validation models (Average Val Acc: {avg_kfold_val_acc:.4f}) performed better than Initial MobileNetV2 (Test Acc: {mobilenetv2_initial_acc:.4f}). Saving best K-Fold model.")
    shutil.copy(best_kfold_model_path, best_overall_model_path) 

logging.info(f"The best performing model is saved to: {best_overall_model_path}")

# for model_path in fold_models_paths:
#     try:
#         os.remove(model_path)
#     except Exception as e:
#         logging.warning(f"Error deleting model file {model_path}: {e}")
# for fold in range(NUM_FOLDS):
#     try:
#         shutil.rmtree(f'hyperband_fold_{fold}')
#     except Exception as e:
#         logging.warning(f"Error deleting hyperband directory for fold {fold}: {e}")
