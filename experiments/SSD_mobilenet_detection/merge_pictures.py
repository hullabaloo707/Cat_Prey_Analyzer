from PIL import Image, ImageDraw
import os

# Load the images
image_path = os.path.join("data","cat_data_set_eval","00000005_000.jpg")
mouse_image = os.path.join("data","mouse.png")

background = Image.open(image_path).convert("RGBA")
foreground = Image.open(mouse_image).convert("RGBA")

# Get the size of the background image
width, height = background.size
width_mouse = int(width/10)
height_mouse = int(height/10)

# Create a circular mask
mask = Image.new('L', (width_mouse, height_mouse), 0)
draw = ImageDraw.Draw(mask)
draw.ellipse((0, 0, width_mouse, height_mouse), fill=255)

# Resize the foreground image to fit inside the background
foreground = foreground.resize((width_mouse,height_mouse))
foreground = foreground.rotate(45)

foreground.putalpha(mask)
im = Image.composite(im1, im2, mask)

# Get the size of the foreground image
fw, fh = foreground.size

# Calculate the position to place the foreground image in the center of the background
x = int((width - fw) / 2)
y = int((height - fh) / 2)

# Paste the foreground image onto the background
background.paste(foreground, (x, y), foreground)

# Save the result
background.save('result.png')