from PIL import Image
import pathlib
import os
image = os.path.join("../..","debug","input","08-20201012200254-03.jpg")



PATH_TO_TEST_IMAGES_DIR = pathlib.Path('../../debug/input/') #models/research/object_detection/test_images
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
output_path = os.path.join("../..","debug","output","input_cropped")


for image_path in TEST_IMAGE_PATHS:

    # Open the original image
    img = Image.open(image_path)

    # Calculate the coordinates for the three cropped images
    x1 = 0
    y1 = 720 - 380
    x2 = 380
    y2 = 720
    overlap = 90

    # Crop and save the three images
    for i in range(3):
        cropped = img.crop((x1, y1, x2, y2))
        cropped.save(os.path.join(output_path,f"{image_path.stem}_{i}.jpg"))
        x1 += (380 - overlap)
        x2 += (380 - overlap)