#Kalman Filter
import cv2
import numpy as np
import math

collision_coordinates = [[2,3], [1,7], []]

def lat_to_cart(lat, long):
    x = long * 60 * 1852 * math.cos(lat)
    y = lat * 60 * 1852
    x = x * 0.175
    y = y * 0.175
    print(x,y)

#method of
def coordinate_mapping():
    pass

img = cv2.imread('test.jpg', 0)
edges = cv2.Canny(img, 100, 255)

indices = np.where(edges != [0])
coordinates = zip(indices[0], indices[1])

xcoord = indices[0]
ycoords = indices[1]

print(indices)
print(xcoord,ycoords )

