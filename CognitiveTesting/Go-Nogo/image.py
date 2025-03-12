from PIL import Image
import os

def add_blank_space(folder_path, output_folder, blank_height=80):
    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            
            # Create a new image with extra blank space at the bottom
            new_height = img.height + blank_height
            new_img = Image.new('RGBA', (img.width, new_height), (0, 0, 0, 0))  # Transparent background
            new_img.paste(img, (0, 0))  # Paste the original image onto the new image
            
            # Save the new image
            new_img.save(os.path.join(output_folder, filename))
            print(f"Processed: {filename}")

# Add blank space to images in the TopLeft folder
add_blank_space('/Users/jaeyoungkang/Documents/BME/4A/BME461/BeatSlasher/assets/sprites/TopLeft', '/Users/jaeyoungkang/Documents/BME/4A/BME461/BeatSlasher/assets/sprites/TopLeft')

# Add blank space to images in the TopRight folder
add_blank_space('/Users/jaeyoungkang/Documents/BME/4A/BME461/BeatSlasher/assets/sprites/TopRight', '/Users/jaeyoungkang/Documents/BME/4A/BME461/BeatSlasher/assets/sprites/TopRight')
