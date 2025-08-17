import numpy as np
import cv2
import os
from torchvision import transforms
import shutil
import onnx
import onnxruntime as rt
from PIL import Image

def get_transforms(image_size):
    trans = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return trans

def main():
    # Hardcoded paths
    model_path = r"C:\Users\dhc\Desktop\i2v_task\EfficientNet\efficientnet_b0_best.onnx"
    classes_path = r"C:\Users\dhc\Desktop\i2v_task\sample_classes.txt"
    image_size = (299,299)  # Square image size
    verification_dir = r"C:\Users\dhc\Desktop\i2v_task\verification_images"
    output_dir = r"C:\Users\dhc\Desktop\i2v_task\EfficientNet\verify"

    # Validate inputs
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.isfile(classes_path):
        raise FileNotFoundError(f"Classes file not found: {classes_path}")

    # Setup ONNX
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    sess = rt.InferenceSession(model_path)

    # Input and output details
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    print(f"Input name: {input_name}, Shape: {input_shape}")

    output_name = sess.get_outputs()[0].name
    output_shape = sess.get_outputs()[0].shape
    print(f"Output name: {output_name}, Shape: {output_shape}")

    # Get transforms
    trans = get_transforms(image_size)

    # Load classes (strip newlines)
    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    print(f"Classes: {classes}, Number of classes: {len(classes)}")

    # Create output directories
    org_dir = os.path.join(output_dir, "org")
    pred_dir = os.path.join(output_dir, "pred")
    os.makedirs(org_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    # Process verification images
    for label in ["autorickshaw", "car", "van"]:
        image_filename = f"{label}.png"
        image_filepath = os.path.join(verification_dir, image_filename)
        org_file_path = os.path.join(org_dir, image_filename)

        # Copy image if not already present
        if not os.path.isfile(image_filepath):
            print(f"Warning: {image_filepath} not found. Skipping.")
            continue
        if not os.path.isfile(org_file_path):
            shutil.copy(image_filepath, org_file_path)

        # Load image
        org_image = cv2.imread(org_file_path)
        if org_image is None:
            print(f"Error: Failed to load {org_file_path}. Skipping.")
            continue

        # Convert to RGB and then to PIL Image
        image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        # Apply transforms
        image_tensor = trans(image).unsqueeze(0)
        image_np = image_tensor.numpy().astype(np.float32)

        # Predict
        pred_outs = sess.run(None, {input_name: image_np})[0][0]
        pred_arg = pred_outs.argmax()
        pred_conf = pred_outs[pred_arg]
        pred_cls = classes[pred_arg]
        pred_text = f"{pred_cls} {pred_conf:.2f}"

        # Print prediction
        print(f"Prediction for {image_filename}: {pred_text}")

        # Add prediction to image
        cv2.putText(
            org_image,
            pred_text,
            (25, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Save predicted image
        save_path = os.path.join(pred_dir, f"pred_{image_filename}")
        cv2.imwrite(save_path, org_image)
        print(f"Image prediction successfully saved to {save_path}")

if __name__ == "__main__":
    main()