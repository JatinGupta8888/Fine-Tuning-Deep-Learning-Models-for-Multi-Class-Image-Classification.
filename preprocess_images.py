import os
import shutil
from cleanvision import Imagelab

# Set random seed for reproducibility
import numpy as np
np.random.seed(42)

# Directories
DATA_DIR = r"C:\Users\dhc\Desktop\i2v_task\vehicle_dataset"  # Adjust this to your actual path if needed
TRAIN_DIR = os.path.join(DATA_DIR, "train")
OUTPUT_DIR = r"C:\Users\dhc\Desktop\i2v_task\Cleaned"  # Where cleaned data and outputs will be saved
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function for Data Analysis and Cleaning with Cleanvision
def analyze_and_clean_data():
    print("Starting data cleaning...")
    # Initialize Imagelab with the training data directory
    imagelab = Imagelab(data_path=TRAIN_DIR)
    
    # Find all default issues (includes blurry, exact_duplicates, etc.)
    imagelab.find_issues()
    
    # Generate a report of identified issues
    imagelab.report()
    
    # Define the directory for cleaned data
    cleaned_dir = os.path.join(OUTPUT_DIR, "cleaned_train")
    os.makedirs(cleaned_dir, exist_ok=True)
    
    # Identify problematic files (duplicates and blurry images)
    duplicate_files = set()
    blurry_files = set()
    
    # Check for exact duplicates
    if "exact_duplicates" in imagelab.issues.columns:
        duplicate_sets = imagelab.issues[imagelab.issues["is_exact_duplicates_issue"]]["exact_duplicates_set"].dropna()
        for duplicate_set in duplicate_sets:
            files = list(duplicate_set)
            # Keep the first file, mark others as duplicates to exclude
            duplicate_files.update(files[1:])
    
    # Check for blurry images
    if "is_blurry_issue" in imagelab.issues.columns:
        blurry_files = set(imagelab.issues[imagelab.issues["is_blurry_issue"]].index)
    
    # Combine all files to exclude
    excluded_files = duplicate_files.union(blurry_files)
    
    # Copy non-problematic files to the cleaned directory
    for root, _, files in os.walk(TRAIN_DIR):
        rel_path = os.path.relpath(root, TRAIN_DIR)
        target_dir = os.path.join(cleaned_dir, rel_path)
        os.makedirs(target_dir, exist_ok=True)
        for file in files:
            src_path = os.path.join(root, file)
            if src_path not in excluded_files:
                shutil.copy(src_path, target_dir)
                print(f"Copied clean image: {src_path} to {target_dir}")
            else:
                print(f"Excluded problematic image: {src_path}")
    
    print(f"Data cleaning complete. Cleaned data saved to: {cleaned_dir}")
    return cleaned_dir

if __name__ == "__main__":
    # Run the data analysis and cleaning
    Cleaned_Train = analyze_and_clean_data()