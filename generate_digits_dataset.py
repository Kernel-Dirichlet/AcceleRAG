import os
import sys
import numpy as np
from sklearn.datasets import load_digits
from PIL import Image
import matplotlib.pyplot as plt

def save_digit_image(digit_data, label, index, output_dir):
    """Save a single digit image to the appropriate directory."""
    # Reshape the 1D array into 2D (8x8)
    digit_image = digit_data.reshape(8, 8)
    
    # Scale to 0-255 range
    digit_image = (digit_image * 255).astype(np.uint8)
    
    # Create directory if it doesn't exist
    digit_dir = os.path.join(output_dir, f'digit_{label}')
    os.makedirs(digit_dir, exist_ok=True)
    
    # Save the image
    image_path = os.path.join(digit_dir, f'digit_{label}_{index:03d}.png')
    Image.fromarray(digit_image).save(image_path)
    
    return image_path

def generate_digits_dataset(output_dir='digits_dataset'):
    """Generate a directory structure for the digits dataset."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the digits dataset
    digits = load_digits()
    images = digits.images
    labels = digits.target
    
    # Create a visualization of one example of each digit
    plt.figure(figsize=(10, 2))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.imshow(images[labels == i][0], cmap='gray')
        plt.axis('off')
        plt.title(f'Digit {i}')
    
    # Save the visualization
    plt.savefig(os.path.join(output_dir, 'sample_digits.png'))
    plt.close()
    
    # Process each digit
    for label in range(10):
        # Get all images for this digit
        digit_indices = np.where(labels == label)[0]
        digit_images = images[digit_indices]
        
        print(f"Processing digit {label} with {len(digit_images)} samples...")
        
        # Save each image
        for i, img_data in enumerate(digit_images):
            save_digit_image(img_data, label, i, output_dir)
            
    print(f"Dataset generated in {output_dir}")
    print(f"Total samples: {len(images)}")
    print(f"Samples per digit: {len(digit_images)}")

if __name__ == '__main__':
    generate_digits_dataset() 
