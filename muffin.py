{
 "nbformat": 4,
 "nbformat_minor": 0,
 "metadata": {
  "colab": {
   "provenance": [],
   "gpuType": "T4"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  },
  "accelerator": "GPU"
 },
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Muffin vs Chihuahua Binary Image Classification \u2014 CNN & VGG16\n### COMP 20037 \u2014 Artificial Intelligence and Deep Learning\n### Student: Said alkiyumi | ID: 22f23129\n---\n**Problem:** Binary classification \u2014 Muffin vs Chihuahua\n**Dataset:** Muffin vs Chihuahua \u2014 Kaggle (5,917 images)\n**Models:** Custom CNN | VGG16 Transfer Learning"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 1: Mount Google Drive"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Mount Google Drive to access the dataset\nfrom google.colab import drive\ndrive.mount('/content/drive')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 2: Import Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import all required libraries\nimport os\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport pandas as pd\n\nimport tensorflow as tf\nfrom tensorflow.keras import layers, models\nfrom tensorflow.keras.applications import VGG16\nfrom tensorflow.keras.preprocessing.image import ImageDataGenerator\nfrom tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau\nfrom sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score\n\n# Verify GPU\nprint('TensorFlow:', tf.__version__)\nprint('GPU:', tf.config.list_physical_devices('GPU'))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 3: Verify Dataset Path in Google Drive"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Dataset already exists in Google Drive \u2014 no download needed\nDRIVE_TRAIN = '/content/drive/MyDrive/Muffin vs chihuahua/train'\nDRIVE_TEST  = '/content/drive/MyDrive/Muffin vs chihuahua/test'\n\n# Verify folders and count images\nfor split, path in [('Train', DRIVE_TRAIN), ('Test', DRIVE_TEST)]:\n    classes = [c for c in os.listdir(path) if os.path.isdir(os.path.join(path, c))]\n    print(f'{split}: {classes}')\n    for cls in classes:\n        count = len(os.listdir(os.path.join(path, cls)))\n        print(f'  {cls}: {count} images')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 3b: Copy Dataset to Local Storage (Faster Training)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Copy dataset from Google Drive to Colab local storage for faster training\n",
    "# Reading directly from Drive is slow \u2014 local storage is much faster\n",
    "import shutil, os\n",
    "\n",
    "LOCAL_BASE = '/content/muffin_data'\n",
    "\n",
    "if not os.path.exists(LOCAL_BASE):\n",
    "    print('Copying dataset to local storage (one-time, ~1-2 minutes)...')\n",
    "    shutil.copytree('/content/drive/MyDrive/Muffin vs chihuahua/train', f'{LOCAL_BASE}/train')\n",
    "    shutil.copytree('/content/drive/MyDrive/Muffin vs chihuahua/test',  f'{LOCAL_BASE}/test')\n",
    "    print('Done! Dataset copied to local storage.')\n",
    "else:\n",
    "    print('Dataset already in local storage.')\n",
    "\n",
    "# Update paths to local storage\n",
    "DRIVE_TRAIN = f'{LOCAL_BASE}/train'\n",
    "DRIVE_TEST  = f'{LOCAL_BASE}/test'\n",
    "print(f'Train path: {DRIVE_TRAIN}')\n",
    "print(f'Test path:  {DRIVE_TEST}')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 4: Data Preprocessing and Augmentation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "IMG_SIZE   = (128, 128)\nBATCH_SIZE = 32\n\n# Training generator with data augmentation to prevent overfitting\ntrain_datagen = ImageDataGenerator(\n    rescale=1.0/255,          # Normalize pixel values to [0, 1]\n    rotation_range=20,        # Random rotation up to 20 degrees\n    width_shift_range=0.2,    # Random horizontal shift\n    height_shift_range=0.2,   # Random vertical shift\n    horizontal_flip=True,     # Random horizontal flip\n    zoom_range=0.2,           # Random zoom in/out\n    validation_split=0.2      # Reserve 20% for validation\n)\n\n# Test generator \u2014 normalization only\ntest_datagen = ImageDataGenerator(rescale=1.0/255)\n\ntrain_gen = train_datagen.flow_from_directory(\n    DRIVE_TRAIN,\n    target_size=IMG_SIZE, batch_size=BATCH_SIZE,\n    class_mode='binary', subset='training', seed=42\n)\nval_gen = train_datagen.flow_from_directory(\n    DRIVE_TRAIN,\n    target_size=IMG_SIZE, batch_size=BATCH_SIZE,\n    class_mode='binary', subset='validation', seed=42\n)\ntest_gen = test_datagen.flow_from_directory(\n    DRIVE_TEST,\n    target_size=IMG_SIZE, batch_size=BATCH_SIZE,\n    class_mode='binary', shuffle=False\n)\n\nprint('Class indices:', train_gen.class_indices)\nprint(f'Training: {train_gen.samples} | Validation: {val_gen.samples} | Test: {test_gen.samples}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 5: Visualize Sample Images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.image as mpimg\n\nclasses = list(train_gen.class_indices.keys())\nfig, axes = plt.subplots(2, 4, figsize=(14, 7))\nfig.suptitle('Sample Images \u2014 Muffin vs Chihuahua', fontsize=13, fontweight='bold')\n\nfor row, cls in enumerate(classes):\n    folder = os.path.join(DRIVE_TRAIN, cls)\n    imgs = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))][:4]\n    for col, name in enumerate(imgs):\n        img = mpimg.imread(os.path.join(folder, name))\n        axes[row][col].imshow(img)\n        axes[row][col].set_title(cls, fontsize=9)\n        axes[row][col].axis('off')\n\nplt.tight_layout()\nplt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 6: Basic CNN (Baseline \u2014 No Regularization to Demonstrate Overfitting)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Basic CNN WITHOUT BatchNormalization or Dropout\n# Purpose: demonstrate overfitting clearly\ndef build_basic_cnn():\n    model = models.Sequential([\n        # Block 1: 32 filters\n        layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),\n        layers.MaxPooling2D(2,2),\n        # Block 2: 64 filters\n        layers.Conv2D(64, (3,3), activation='relu'),\n        layers.MaxPooling2D(2,2),\n        # Block 3: 128 filters\n        layers.Conv2D(128, (3,3), activation='relu'),\n        layers.MaxPooling2D(2,2),\n        # Fully connected \u2014 no dropout\n        layers.Flatten(),\n        layers.Dense(256, activation='relu'),\n        layers.Dense(1, activation='sigmoid')   # Binary output\n    ], name='Basic_CNN')\n    return model\n\nbasic_cnn = build_basic_cnn()\nbasic_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\nbasic_cnn.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train Basic CNN \u2014 no callbacks to show overfitting\nbasic_history = basic_cnn.fit(\n    train_gen,\n    validation_data=val_gen,\n    epochs=15,\n    verbose=1\n)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot Basic CNN curves \u2014 shows overfitting gap\nfig, axes = plt.subplots(1, 2, figsize=(14, 5))\nfig.suptitle('Basic CNN \u2014 Overfitting Demo (Training vs Validation)', fontsize=12)\naxes[0].plot(basic_history.history['accuracy'],     label='Train', color='blue')\naxes[0].plot(basic_history.history['val_accuracy'], label='Val',   color='orange')\naxes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch'); axes[0].legend()\naxes[1].plot(basic_history.history['loss'],     label='Train', color='blue')\naxes[1].plot(basic_history.history['val_loss'], label='Val',   color='orange')\naxes[1].set_title('Loss'); axes[1].set_xlabel('Epoch'); axes[1].legend()\nplt.tight_layout(); plt.show()\n\n# Evaluate on test set\nbasic_preds = (basic_cnn.predict(test_gen) > 0.5).astype(int).flatten()\nprint('Basic CNN Classification Report:')\nprint(classification_report(test_gen.classes, basic_preds,\n                             target_names=list(train_gen.class_indices.keys())))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 7: Custom CNN with BatchNorm + Dropout (Regularized)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Custom CNN WITH BatchNormalization and Dropout to prevent overfitting\ndef build_custom_cnn():\n    model = models.Sequential([\n        # Block 1: 32 filters\n        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(128,128,3)),\n        layers.BatchNormalization(),   # Normalize activations\n        layers.MaxPooling2D(2,2),\n        layers.Dropout(0.25),          # Drop 25% of neurons\n        # Block 2: 64 filters\n        layers.Conv2D(64, (3,3), activation='relu', padding='same'),\n        layers.BatchNormalization(),\n        layers.MaxPooling2D(2,2),\n        layers.Dropout(0.25),\n        # Block 3: 128 filters\n        layers.Conv2D(128, (3,3), activation='relu', padding='same'),\n        layers.BatchNormalization(),\n        layers.MaxPooling2D(2,2),\n        layers.Dropout(0.25),\n        # Fully connected\n        layers.Flatten(),\n        layers.Dense(256, activation='relu'),\n        layers.BatchNormalization(),\n        layers.Dropout(0.5),           # Heavier dropout before output\n        layers.Dense(1, activation='sigmoid')\n    ], name='Custom_CNN')\n    return model\n\ncustom_cnn = build_custom_cnn()\ncustom_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\ncustom_cnn.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Callbacks to prevent overfitting\ncallbacks = [\n    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),\n    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)\n]\n\n# Train Custom CNN\ncustom_history = custom_cnn.fit(\n    train_gen,\n    validation_data=val_gen,\n    epochs=30,\n    callbacks=callbacks,\n    verbose=1\n)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot Custom CNN curves\nfig, axes = plt.subplots(1, 2, figsize=(14, 5))\nfig.suptitle('Custom CNN \u2014 With Regularization', fontsize=12)\naxes[0].plot(custom_history.history['accuracy'],     label='Train', color='blue')\naxes[0].plot(custom_history.history['val_accuracy'], label='Val',   color='orange')\naxes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch'); axes[0].legend()\naxes[1].plot(custom_history.history['loss'],     label='Train', color='blue')\naxes[1].plot(custom_history.history['val_loss'], label='Val',   color='orange')\naxes[1].set_title('Loss'); axes[1].set_xlabel('Epoch'); axes[1].legend()\nplt.tight_layout(); plt.show()\n\n# Evaluate\ncustom_preds = (custom_cnn.predict(test_gen) > 0.5).astype(int).flatten()\ntrue_labels  = test_gen.classes\nclass_names  = list(train_gen.class_indices.keys())\nprint('Custom CNN Classification Report:')\nprint(classification_report(true_labels, custom_preds, target_names=class_names))\n\ncm = confusion_matrix(true_labels, custom_preds)\nplt.figure(figsize=(6,5))\nsns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n            xticklabels=class_names, yticklabels=class_names)\nplt.title('Custom CNN Confusion Matrix')\nplt.ylabel('True'); plt.xlabel('Predicted')\nplt.tight_layout(); plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 8: VGG16 Transfer Learning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load VGG16 pre-trained on ImageNet \u2014 freeze base layers\nbase_model = VGG16(weights='imagenet', include_top=False, input_shape=(128,128,3))\nbase_model.trainable = False\nprint(f'VGG16 base: {len(base_model.layers)} layers frozen')\n\n# Add custom classification head\nvgg16_model = models.Sequential([\n    base_model,\n    layers.Flatten(),\n    layers.Dense(256, activation='relu'),\n    layers.BatchNormalization(),\n    layers.Dropout(0.5),\n    layers.Dense(128, activation='relu'),\n    layers.BatchNormalization(),\n    layers.Dropout(0.3),\n    layers.Dense(1, activation='sigmoid')    # Binary output\n], name='VGG16_Transfer')\n\nvgg16_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\nvgg16_model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train VGG16\nvgg16_history = vgg16_model.fit(\n    train_gen,\n    validation_data=val_gen,\n    epochs=20,\n    callbacks=callbacks,\n    verbose=1\n)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot VGG16 curves\nfig, axes = plt.subplots(1, 2, figsize=(14, 5))\nfig.suptitle('VGG16 Transfer Learning', fontsize=12)\naxes[0].plot(vgg16_history.history['accuracy'],     label='Train', color='blue')\naxes[0].plot(vgg16_history.history['val_accuracy'], label='Val',   color='orange')\naxes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch'); axes[0].legend()\naxes[1].plot(vgg16_history.history['loss'],     label='Train', color='blue')\naxes[1].plot(vgg16_history.history['val_loss'], label='Val',   color='orange')\naxes[1].set_title('Loss'); axes[1].set_xlabel('Epoch'); axes[1].legend()\nplt.tight_layout(); plt.show()\n\n# Evaluate VGG16\nvgg16_preds = (vgg16_model.predict(test_gen) > 0.5).astype(int).flatten()\nprint('VGG16 Classification Report:')\nprint(classification_report(true_labels, vgg16_preds, target_names=class_names))\n\ncm_vgg = confusion_matrix(true_labels, vgg16_preds)\nplt.figure(figsize=(6,5))\nsns.heatmap(cm_vgg, annot=True, fmt='d', cmap='Greens',\n            xticklabels=class_names, yticklabels=class_names)\nplt.title('VGG16 Confusion Matrix')\nplt.ylabel('True'); plt.xlabel('Predicted')\nplt.tight_layout(); plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 9: Final Model Comparison Table"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Compare all three models\ndef get_metrics(true, pred, name):\n    return {\n        'Model':     name,\n        'Accuracy':  f\"{accuracy_score(true, pred)*100:.2f}%\",\n        'Precision': f\"{precision_score(true, pred)*100:.2f}%\",\n        'Recall':    f\"{recall_score(true, pred)*100:.2f}%\",\n        'F1-Score':  f\"{f1_score(true, pred)*100:.2f}%\"\n    }\n\nresults = [\n    get_metrics(true_labels, basic_preds,  'Basic CNN (No Regularization)'),\n    get_metrics(true_labels, custom_preds, 'Custom CNN (BatchNorm + Dropout)'),\n    get_metrics(true_labels, vgg16_preds,  'VGG16 Transfer Learning')\n]\n\ndf = pd.DataFrame(results)\nprint('\\n============ MODEL COMPARISON TABLE ============')\nprint(df.to_string(index=False))\n\n# Bar chart\nmodel_names = ['Basic CNN', 'Custom CNN', 'VGG16']\nacc_vals = [accuracy_score(true_labels, p)*100 for p in [basic_preds, custom_preds, vgg16_preds]]\nrec_vals = [recall_score(true_labels, p)*100   for p in [basic_preds, custom_preds, vgg16_preds]]\n\nx = np.arange(len(model_names))\nfig, ax = plt.subplots(figsize=(9, 5))\nbars1 = ax.bar(x - 0.2, acc_vals, 0.35, label='Accuracy', color='steelblue')\nbars2 = ax.bar(x + 0.2, rec_vals, 0.35, label='Recall',   color='darkorange')\nax.set_ylabel('Score (%)'); ax.set_title('Model Comparison: Accuracy vs Recall')\nax.set_xticks(x); ax.set_xticklabels(model_names); ax.set_ylim(50, 110); ax.legend()\nfor b in list(bars1) + list(bars2):\n    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,\n            f'{b.get_height():.1f}%', ha='center', fontsize=8)\nplt.tight_layout(); plt.show()\nprint('\\nAll done! Copy screenshots of outputs to paste into your report.')"
   ]
  }
 ]
}
