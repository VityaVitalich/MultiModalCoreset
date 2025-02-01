from PIL import Image
import numpy as np
import os
from tqdm import tqdm


def image_stats(image_path):
    """Read an image and return its statistics (mean and squared values)."""
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)  # / 255.0  # Normalize pixel values to [0, 1]
    mean = np.mean(image_np, axis=(0, 1))  # Channel-wise mean
    std = np.std(image_np, axis=(0, 1))  # Squared mean for std calculation
    return (
        mean,
        std,
    )  # , image_np.size / 3  # Return the mean, squared mean, and number of pixels (per channel)


def update_running_stats(stats, new_data):
    """Update running means and squared means given new data."""
    (running_mean, running_sq_mean, total_pixels), (mean, sq_mean, pixels) = (
        stats,
        new_data,
    )
    total_pixels += pixels
    running_mean = (
        running_mean * (total_pixels - pixels) + mean * pixels
    ) / total_pixels
    running_sq_mean = (
        running_sq_mean * (total_pixels - pixels) + sq_mean * pixels
    ) / total_pixels
    return running_mean, running_sq_mean, total_pixels


def calculate_overall_stats(directory):
    total_mean = np.zeros(3)
    total_std = np.zeros(3)
    total_objects = 0

    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(
            (".png", ".jpg", ".jpeg")
        ):  # Add other file extensions if needed.
            filepath = os.path.join(directory, filename)
            mean, std = image_stats(filepath)
            total_mean += mean
            total_std += std
            total_objects += 1

    return total_mean / total_objects, total_std / total_objects


if __name__ == "__main__":
    directory = "/home/data/dq/clevr_complex/train/rgb"  # Update this path
    mean, std = calculate_overall_stats(directory)

    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std}")
