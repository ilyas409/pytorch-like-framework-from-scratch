import numpy as np
import urllib.request
import os

def download_iris_dataset(save_locally=True):
    """
    Download the Iris dataset directly from UCI repository
    Returns binary classification version (Setosa vs others)
    """
    # UCI Iris dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    filename = "iris.data"
    
    print("Downloading Iris dataset from UCI repository...")
    
    try:
        # Download the dataset
        urllib.request.urlretrieve(url, filename)
        print(f"Dataset downloaded successfully as '{filename}'")
        
        # Read the data
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Parse the data
        data = []
        labels = []
        
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                parts = line.split(',')
                if len(parts) == 5:  # Should have 4 features + 1 label
                    # Extract features (first 4 columns)
                    features = [float(x) for x in parts[:4]]
                    # Extract label (last column)
                    label = parts[4]
                    
                    data.append(features)
                    labels.append(label)
        
        # Convert to numpy arrays
        X = np.array(data, dtype=np.float32)
        
        # Convert labels to binary (Setosa vs others)
        y_binary = np.array([1.0 if label == 'Iris-setosa' else 0.0 for label in labels], dtype=np.float32)
        y_binary = y_binary.reshape(-1, 1)
        
        # Standardize features (zero mean, unit variance)
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_standardized = (X - X_mean) / X_std
        
        # Clean up file if not saving locally
        if not save_locally and os.path.exists(filename):
            os.remove(filename)
            print("Temporary file cleaned up.")
        
        print(f"Dataset loaded successfully!")
        print(f"Features shape: {X_standardized.shape}")
        print(f"Labels shape: {y_binary.shape}")
        print(f"Feature names: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']")
        print(f"Classes: Setosa (1) vs Others (0)")
        print(f"Class distribution: {np.bincount(y_binary.flatten().astype(int))}")
        
        return X_standardized, y_binary, labels
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None, None, None

def download_iris_multiclass(save_locally=True):
    """
    Download the full 3-class Iris dataset
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    filename = "iris.data"
    
    print("Downloading full Iris dataset (3 classes)...")
    
    try:
        urllib.request.urlretrieve(url, filename)
        
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        data = []
        labels = []
        
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) == 5:
                    features = [float(x) for x in parts[:4]]
                    label = parts[4]
                    data.append(features)
                    labels.append(label)
        
        X = np.array(data, dtype=np.float32)
        
        # Create label mapping
        unique_labels = list(set(labels))
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        y_multiclass = np.array([label_to_int[label] for label in labels], dtype=np.float32)
        y_multiclass = y_multiclass.reshape(-1, 1)
        
        # Standardize features
        X_standardized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
        if not save_locally and os.path.exists(filename):
            os.remove(filename)
        
        print(f"Full dataset loaded!")
        print(f"Features shape: {X_standardized.shape}")
        print(f"Labels shape: {y_multiclass.shape}")
        print(f"Classes: {unique_labels}")
        
        return X_standardized, y_multiclass, unique_labels
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None, None, None

def create_custom_binary_split(X, y_original_labels, class1, class2):
    """
    Create a custom binary classification from the 3-class iris dataset
    """
    # Find indices for the two classes we want
    indices = []
    binary_labels = []
    
    for i, label in enumerate(y_original_labels):
        if label == class1:
            indices.append(i)
            binary_labels.append(0.0)
        elif label == class2:
            indices.append(i)
            binary_labels.append(1.0)
    
    X_binary = X[indices]
    y_binary = np.array(binary_labels, dtype=np.float32).reshape(-1, 1)
    
    print(f"Created binary classification: {class1} (0) vs {class2} (1)")
    print(f"Dataset shape: {X_binary.shape}")
    print(f"Class distribution: {np.bincount(y_binary.flatten().astype(int))}")
    
    return X_binary, y_binary

