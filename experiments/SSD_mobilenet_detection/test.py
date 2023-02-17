from PIL import Image, ImageDraw, ImageOps
import os
# Open the image
mouse_image = os.path.join("data","mouse.png")

image = Image.open(mouse_image)

# # Create a mask with the same size as the image
# mask = Image.new('L', image.size, 0)
#
# # Create a draw object for the mask
# draw = ImageDraw.Draw(mask)
#
# # Draw a rounded rectangle over the mask
# draw.rounded_rectangle((0, 0, image.size[0], image.size[1]), fill=0, radius=450)
#
#
mask = Image.new("L", image.size, 0)
draw = ImageDraw.Draw(mask)
# draw.ellipse((140, 50, 260, 170), fill=255)
draw.rounded_rectangle((200, 200, image.size[0]-200, image.size[1]-200), fill=255, radius=50)


# mask = Image.new("L", image.size, 0)
# draw = ImageDraw.Draw(mask)
# draw.ellipse((140, 50, 260, 170), fill=255)

# Create a new transparent image with the same size as the original image
background = Image.new('RGBA', image.size, (0, 0, 0, 0))
im = Image.composite(image,background,mask=mask)

# Save the result
im.save('example_result.png')