from PIL import Image, ImageDraw
import os
import numpy as np
import math

# Load the images
cat_data_set_dir = os.path.join("data","cat_data_set_eval")
cat_path = os.path.join(cat_data_set_dir,"00000005_000.jpg")
mouse_path = os.path.join("data","mouses_cropped","mouse-0.png")

def merge_cat_and_mouse(cat_path,mouse_path,angle):
    background = Image.open(cat_path).convert("RGBA")
    foreground = Image.open(mouse_path).convert("RGBA")
    foreground = foreground.rotate(-90,expand=True)


    if cat_path.endswith(".jpg"):
        # print(file)
        file_annotation = cat_path + ".cat"
        # if not os.path.exists(file_annotation):
        #     print(f"skip: {image_path}")
        #     continue

        with open( file_annotation, "r") as features_file:
            annotations = features_file.readline().rstrip()
            annotations = annotations.split(" ")
            annotations = [int(i) for i in annotations]

            left_eye = np.array([annotations[2],annotations[1]])
            right_eye = np.array([annotations[4],annotations[3]])
            mouth = np.array([annotations[6],annotations[5]])
            left_ear = np.array([annotations[10],annotations[9]])
            right_ear = np.array([annotations[16],annotations[15]])

            eye_vec = np.subtract(left_eye,right_eye)
            distance_eyes = math.sqrt(math.pow(eye_vec[0],2)+math.pow(eye_vec[1],2))


    # Get the size of the background image
    width, height = background.size
    width_mouse = int(width/10)
    height_mouse = int(height/10)

    foreground = foreground.rotate(10,center=(int(foreground.size[0]/2),10),expand=True)

    foreground.size
    # Resize the foreground image to fit inside the background
    foreground = foreground.resize((int(foreground.size[0]*distance_eyes/background.size[0]),int(foreground.size[1]*distance_eyes/background.size[1])))


    # Get the size of the foreground image
    fw, fh = foreground.size

    # Calculate the position to place the foreground image in the center of the background
    x = int((width - fw) / 2)
    y = int((height - fh) / 2)

    # Paste the foreground image onto the background
    background.paste(foreground, (mouth[1]-int(foreground.size[0]/2),int(mouth[0]-distance_eyes/10)),foreground)

# Save the result
background.save('result.png')