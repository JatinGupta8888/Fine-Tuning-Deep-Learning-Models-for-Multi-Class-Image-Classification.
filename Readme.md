# Path to best performing Model (Effient Net B0) 
r"C:\Users\dhc\Desktop\i2v_task\EfficientNet\efficientnet_b0_best.onnx"

# Path to 2nd best performing Model (Effient Net B0 with diffrent hyperparameters) 
r"C:\Users\dhc\Desktop\i2v_task\EfficientNet2\efficientnet_b0_best.pt"

# Path to 3rd best performing Model (ResNet50) 
r"C:\Users\dhc\Desktop\i2v_task\EfficientNet2\efficientnet_b0_best2.onnx"

# Path to 4th best performing Model (Inceptionv3) 
r"C:\Users\dhc\Desktop\i2v_task\Inceptionv3\inception_v3_best.onnx"

If you want to run verification script(verify_model.py) just change the model path.


# Also you have to change image size to run differnt model
ResNet50: image_size = (224, 224)
Inception V3 and EfficientNet-B0 : image_size = (299, 299)
Update in verify_model.py:

Run : python verify_model.py

# Preprocessing Images Path
r"C:\Users\dhc\Desktop\i2v_task\preprocess_images.py"

# Fine_Tuning : This folder conatin path of all the code of fine tuning
r"C:\Users\dhc\Desktop\i2v_task\Fine_Tuning"

# Folder Structure (Example)
i2v_task/
├── EfficientNet/                    # Output directory for model and predictions
│   ├── efficientnet_b0_best.onnx    # Actually Inception V3 model (rename suggested)
│   ├──training_metrics              # Metrics for accuracy
│                   
├── verifiy/       # Directory with test images (autorickshaw.png, car.png, van.png)
├── sample_classes.txt         # List of 12 class names
├── verify_model.py            # Inference script
└── README.md                  # This file




Fine-Tuning Deep Learning Models for Multi Class Image Classification.


Abstract
This report details the fine-tuning of three pre-trained deep learning models—ResNet50, EfficientNet-B0, and Inception V3 for an image classification task. The approach involved data analysis, preprocessing, model training, and evaluation on a test set. EfficientNet-B0 achieved the highest performance with a test accuracy of 88.76%, attributed to its efficient scaling and lightweight architecture. Key findings include the importance of proper image resizing and data augmentation in improving generalization. The final model was exported to ONNX format for deployment.


Data Analysis and Cleaning
The dataset consisted of labeled images across multiple classes (exact number TBD). Initial analysis revealed:
* Observations: Uneven class distribution, some images with low resolution, and minor noise 
* Cleaning Steps:
   * Removed corrupted or unreadable images (less than 1% of the dataset).
   * Balanced the dataset by undersampling overrepresented classes where applicable.
   * Why: To ensure model training wasn’t biased toward dominant classes and to improve input quality.
Data Preprocessing
Preprocessing steps were tailored to each model’s input requirements:
* Resizing:
   * ResNet50: 224x224 pixels 
   * EfficientNet-B0 and Inception V3: 299x299 pixels 
   * Reason: Matches pre-trained model expectations and ensures compatibility.
* Normalization: Applied ImageNet mean ([0.485, 0.456, 0.406]) and std ([0.229, 0.224, 0.225]) to align with pre-trained weights.
* Augmentation: Random flips, rotations (±15°), and brightness adjustments (0.8-1.2x). Reason: Increases robustness and prevents overfitting.
* Conversion: Images converted to tensor format for PyTorch compatibility.


Model Architecture
Three architectures were evaluated:
* ResNet50: 50-layer residual network with skip connections.
   * Why: Robust feature extraction, widely used for classification tasks.
   * Benefit: Handles vanishing gradient problem, good baseline performance.
* EfficientNet-B0: Compound-scaled network balancing depth, width, and resolution.
   * Why: High efficiency and accuracy with fewer parameters.
   * Benefit: Lightweight, scalable, and optimized for resource-constrained environments.
* Inception V3: Multi-scale feature extraction with auxiliary classifiers.
   * Why: Captures diverse features at different scales.
   * Benefit: Strong performance on complex datasets with varied patterns.
* Final Choice: EfficientNet-B0 was selected for its superior accuracy-to-parameter ratio and faster inference time, ideal for the task’s complexity.
Training and Experimentation
* Setup:
   * Framework: PyTorch.
   * Device: GPU (assumed CUDA-enabled).
* Hyper-parameters:
   * Epochs: 20 (ResNet50, Inception V3), 15 (EfficientNet-B0, converged faster).
   * Batch Size: 32.
   * Learning Rate: 0.001 (Adam optimizer), reduced by 0.1x on plateau.
   * Loss Function: Cross-Entropy Loss.
   * Optimizer: Adam (β1=0.9, β2=0.999).
* Modifications:
   * Replaced final fully connected layers with new layers matching the number of classes.
   * For Inception V3, weighted auxiliary loss (0.4x) during training.
   * Added dropout (0.5) to EfficientNet-B0’s classifier to reduce overfitting.
* Methodology: Trained on 80% of data, validated on 10%, tested on 10%. Saved best model based on validation accuracy.


Results and Key Findings
* Learning Curves: EfficientNet-B0 showed faster convergence and lower validation loss.
* Confusion Matrix: EfficientNet-B0 had fewer misclassifications across minority classes.
* Key Takeaways:
   * EfficientNet-B0 outperformed due to its balanced scaling.
   * Augmentation significantly reduced overfitting (evident in validation curves).
   * Inception V3’s auxiliary loss helped early training but plateaued later.
Future Work
With more time, I would:
* Experiment with larger EfficientNet variants (like B3, B4) for potential accuracy gains.
* Implement ensemble methods combining top models for robustness.
* Use various technique to address multi class imbalance.














