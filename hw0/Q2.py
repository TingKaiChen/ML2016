from PIL import Image
import sys
import numpy

filename = sys.argv[1]

# Read in an image
im = Image.open(filename)
width, height = im.size
imageMat = numpy.array(im)
rotMat = numpy.rot90(imageMat, 2)
newImage = Image.fromarray(rotMat)
newImage.save('ans2.png')