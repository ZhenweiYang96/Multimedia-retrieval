###library and set directory
import os
import sys
import open3d
import pandas as pd
import numpy as np
from math import *


mesh = open3d.io.read_triangle_mesh("data/LabeledDB_new/Human/1.off")
open3d.visualization.draw_geometries([mesh])
###################################################
###Computing volume of a 3D mesh
###First, compute the single volume of the triangle
###Then, compute the whole mesh
###################################################


def signed_volume_triangle(v1, v2, v3):
    v321 = v3[0] * v2[1] * v1[2]
    v231 = v2[0] * v3[1] * v1[2]
    v312 = v3[0] * v1[1] * v2[2]
    v132 = v1[0] * v3[1] * v2[2]
    v213 = v2[0] * v1[1] * v3[2]
    v123 = v1[0] * v2[1] * v3[2]
    return (1.0 / 6.0) * (-v321 + v231 + v312 - v132 - v213 + v123)


def volume_mesh(mesh):

    ###To get the coordination of the points
    vertices = np.asarray(mesh.vertices)
    triangle = np.asarray(mesh.triangles)
    
    volume_sum = 0
    for i in triangle:

        j = i
        coordinates = []

        ###Get coordinates of the points
        for j in i:
            # print("coordinates of the points: ", vertices[j])
            ###List of the coordinates of the three points
            coordinates.append(vertices[j])
            
        volume_sum += signed_volume_triangle(coordinates[0], coordinates[1], coordinates[2])
        #print(coordinates)
    return volume_sum


print(volume_mesh(mesh))







#####################################################################
###calculating the compactness of the mesh
###The equation of compactness is C = I^2 / (4PI * A)
###Firstly, calculating the surface voxels along the boundary (I)
###Then, we calculate the number of pixels inside the segmented shape
#####################################################################

#voxel_grid = open3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=0.5)
#print(voxel_grid)
#open3d.visualization.draw_geometries([voxel_grid])

