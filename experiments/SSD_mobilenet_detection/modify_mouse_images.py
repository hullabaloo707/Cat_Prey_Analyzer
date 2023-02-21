from PIL import Image, ImageDraw, ImageOps
import os
import pathlib
# Open the image
mouse_image = os.path.join("data","mouses","mouse.png")



PATH_TO_TEST_IMAGES_DIR = pathlib.Path('data/mouses') #models/research/object_detection/test_images
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.png")))
OUT_PATH = os.path.join("data","mouses_cropped")


def rotate_and_cut_mouse_image(image_path):
    image = Image.open(image_path)
    # image = image.rotate(rotation,expand=True)


    # Get the bounding box for all non-transparent pixels
    bbox = image.getbbox()
    # Crop the image using the bounding box
    image = image.crop(bbox)

    # # Create a mask with the same size as the image
    mask = Image.new("L", image.size, 0)

    draw = ImageDraw.Draw(mask)
    # draw.ellipse((140, 50, 260, 170), fill=255)
    #draw.rectangle((0, 0, image.size[0], image.size[1]), fill=0)
    draw.ellipse((image.size[0]/3, 0, image.size[0], image.size[1]), fill=255)

    # for x in range(int(image.size[0])):
    #     alpha = int(255 * x/(255/10))
    #     draw.rectangle((image.size[0]/3 + x, 0, image.size[0]/3 +x+1, image.size[1]), fill=alpha)

    # Create a new transparent image with the same size as the original image
    background = Image.new('RGBA', image.size, (0, 0, 0, 0))

    im = Image.composite(image,background,mask=mask)



    bbox = im.getbbox()
    im = im.crop(bbox)

    # Save the result
    im.save(os.path.join(OUT_PATH,f'{os.path.basename(image_path).split(".")[0]}.png'))



for image_path in TEST_IMAGE_PATHS:

    rotate_and_cut_mouse_image(os.path.abspath(image_path))
