import cv2
import numpy as np
from basicpy import BaSiC
import os
import matplotlib.pyplot as plt

def adjust_mean_std(np_images, target_mean, target_std):
    adjusted_images = []
    for img in np_images:
        mean, std = cv2.meanStdDev(img)
        mean = mean[0][0]
        std = std[0][0]
        
        gain = target_std / std
        bias = target_mean - gain * mean
        
        adjusted = img.astype(np.float32)
        
        adjusted = (adjusted - mean) * gain + target_mean
        
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        adjusted_images.append(adjusted)
    
    return adjusted_images

def correct_image(np_image):
    if len(np_image.shape) >= 3:
        blue_channel = np_image[:, :, 0]
        green_channel = np_image[:, :, 1]
        stacked_channels = np.stack([green_channel, blue_channel], axis=0)
    else:
        stacked_channels = np.stack([np_image, np_image], axis=0)

    basic = BaSiC()
    basic.fit(stacked_channels)
    corrected = basic.transform(stacked_channels)

    corrected_image = corrected[1, :, :]
    corrected_image = corrected_image.astype(np.uint8)

    return corrected_image

def balance_local_contrast(image, clip_limit, tile_grid_size):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    return enhanced_bgr



def correct_images(input_folder, output_folder, contrast_balance=False, clip_limit=2, tile_grid_size=8):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    corrected_images = []
    filenames = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".tif"):
            file_path = os.path.join(input_folder, filename)
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            
            filenames.append(filename)
            corrected_image = correct_image(image)
            if contrast_balance:
                corrected_image = balance_local_contrast(corrected_image, clip_limit, (tile_grid_size, tile_grid_size))
            corrected_images.append(corrected_image)
                
    summary_file = os.path.join(output_folder, 'summary.txt')
    with open(summary_file, 'w') as summary_file:
        for i in range(len(filenames)):
            filename = filenames[i]
            image = corrected_images[i]

            output_path = os.path.join(output_folder, f'corrected_{filename}')
            cv2.imwrite(output_path, image)

            mean = np.mean(image)
            std = np.std(image)
            summary_file.write(f'Image: {filename}\n')
            summary_file.write(f'Mean: {mean}\n')
            summary_file.write(f'Std Dev: {std}\n')
            summary_file.write('\n')  

if __name__=="__main__":
    input_folder = '/Users/zxy/iCloud-backup/Documents/BU实习/imagej/VaMiAnalyzer/images'
    output_folder = '/Users/zxy/iCloud-backup/Documents/BU实习/imagej/VaMiAnalyzer/corrected'
    correct_images(input_folder, output_folder)
