import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def segment(image_path):
    # Load the image
    image = cv2.imread(image_path + ".png")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_dir = "segmentation_output/" + image_path[13:] + '/'
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(image_rgb)
    
    os.mkdir(output_dir)
    
    cv2.imwrite(os.path.join(output_dir, "original_image.png"), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    
    # Convert to HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define a precise range for brown shades
    lower_brown = np.array([0, 62, 0])  # Lower bound for brown
    upper_brown = np.array([16, 118, 143])  # Upper bound for brown

    # Create a mask for brown colors
    mask = cv2.inRange(image_hsv, lower_brown, upper_brown)
    
    plt.subplot(2, 3, 2)
    plt.title("Brown Color Mask")
    plt.imshow(mask, cmap="gray")
    
    cv2.imwrite(os.path.join(output_dir, "brown_color_mask.png"), mask)
    
    # Enhance edges using the Canny edge detector
    edges = cv2.Canny(mask, 50, 150)
    
    plt.subplot(2, 3, 3)
    plt.title("Edges")
    plt.imshow(edges, cmap="gray")
    
    cv2.imwrite(os.path.join(output_dir, "edges.png"), edges)

    # Combine the mask and edges for better segmentation
    combined_mask = cv2.bitwise_or(mask, edges)

    # Apply morphological operations to clean and emphasize structure
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    plt.subplot(2, 3, 4)
    plt.title("Morphological Operations")
    plt.imshow(combined_mask, cmap="gray")
    cv2.imwrite(os.path.join(output_dir, "morph.png"), combined_mask)

    # Apply the rectangle mask to all masks
    # Define the rectangle coordinates
    top_left = (883, 963)
    bottom_right = (3697, 5577)

    # Create a mask for the rectangle region
    rectangle_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.rectangle(rectangle_mask, top_left, bottom_right, 255, -1)
    
    plt.subplot(2, 3, 5)
    plt.title("Artifact Filtering")
    plt.imshow(rectangle_mask, cmap="gray")
    cv2.imwrite(os.path.join(output_dir, "crop.png"), rectangle_mask)

    # Apply the rectangle mask to each intermediate mask
    cleaned_mask = cv2.bitwise_and(cleaned_mask, rectangle_mask)
    cv2.imwrite(os.path.join(output_dir, "final_mask.png"), cleaned_mask)

    # Extract the segmented stem from the original image
    segmented_stem = cv2.bitwise_and(image_rgb, image_rgb, mask=cleaned_mask)
    
    plt.subplot(2, 3, 6)
    plt.title("Segmented Stake")
    plt.imshow(segmented_stem)
    cv2.imwrite(os.path.join(output_dir, "segmented_stem.png"), cv2.cvtColor(segmented_stem, cv2.COLOR_RGB2BGR))
    
    # Save all images to the output folder
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure.svg"))
    
    
def main(images):
    for image in images:
        segment("plant_images/" + image)