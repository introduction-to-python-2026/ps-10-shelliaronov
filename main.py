import numpy as np
from skimage.filters import median
from skimage.morphology import ball
from PIL import Image
import image_utils

filename = 'IMG_1050 2.jpg'

image = image_utils.load_image(filename)

clean_image = median(image, ball(3))

edgeMAG = image_utils.edge_detection(clean_image)

threshold = 30
edge_binary = edgeMAG > threshold

edge_image = Image.fromarray((edge_binary * 255).astype(np.uint8))
edge_image.save('my_edges.png')
