import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Define the bounding box coordinates in the format [Top-left x, Top-left y, Width, Height]
bbox_red = (0.103, 0.609, 0.347, 0.32) # new dog
bbox_blue = (0.48, 0.230, 0.51, 0.695)  # cat
bbox_green = (0.00154, 0.15, 0.230, 0.32)  # dog


# Create a figure and axis
fig, ax = plt.subplots(1, figsize=(8, 8))

# Create a white background image
background = np.ones((512, 512, 3))

# Display the background image
ax.imshow(background)

# Create rectangle patches for the bounding boxes
rect_red = patches.Rectangle(
    (bbox_red[0] * 512, bbox_red[1] * 512),
    bbox_red[2] * 512,
    bbox_red[3] * 512,
    linewidth=2,
    edgecolor='r',
    facecolor='none'
)

rect_blue = patches.Rectangle(
    (bbox_blue[0] * 512, bbox_blue[1] * 512),
    bbox_blue[2] * 512,
    bbox_blue[3] * 512,
    linewidth=2,
    edgecolor='b',
    facecolor='none'
)

rect_green = patches.Rectangle(
    (bbox_green[0] * 512, bbox_green[1] * 512),
    bbox_green[2] * 512,
    bbox_green[3] * 512,
    linewidth=2,
    edgecolor='g',
    facecolor='none'
)

# Add the rectangle patches to the plot
ax.add_patch(rect_red)
ax.add_patch(rect_blue)
ax.add_patch(rect_green)

# Save the plot to a file
plt.savefig("bbox_image.png")
plt.close()