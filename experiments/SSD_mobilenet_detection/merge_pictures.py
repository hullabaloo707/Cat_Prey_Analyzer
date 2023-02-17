from PIL import Image
import os

# Load the images
image_path = os.path.join("data","cat_data_set_eval","00000005_000.jpg")
mouse_image = os.path.join("data","mouse.png")

background = Image.open(image_path).convert("RGBA")
foreground = Image.open(mouse_image).convert("RGBA")

# Get the size of the background image
width, height = background.size

# Resize the foreground image to fit inside the background
foreground = foreground.resize((int(width/10), int(height/10)))
foreground = foreground.rotate(45)


# Get the size of the foreground image
fw, fh = foreground.size

# Calculate the position to place the foreground image in the center of the background
x = int((width - fw) / 2)
y = int((height - fh) / 2)

# Paste the foreground image onto the background
background.paste(foreground, (x, y), foreground)

# Save the result
background.save('result.png')