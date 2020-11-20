###library and set directory
import numpy as np
from math import *
from itertools import combinations
import random
import trimesh
from preprocess_function import calc_eigen

#path = "data\\LabeledDB_new\\Human\\1.off"
#mesh = open3d.io.read_triangle_mesh("data\\LabeledDB_new\\Human\\1.off")


###############################
###Get surface area of the mesh
###############################

def calc_mesh_area(mesh):
    mesh_area = mesh.get_surface_area()
    return mesh_area


def calc_convex_surface(path):
    mesh = trimesh.load(path)
    con = mesh.convex_hull
    area = con.area
    print(area)
    return area


##############################
###Axis aligned bounding box
##############################
def calc_bounding_box_volume(mesh):
    bounding_box = mesh.get_axis_aligned_bounding_box()
    bounding_box_volume = bounding_box.volume()
    return bounding_box_volume


####################################################
###Calculate the the longest diameter of the mesh###
####################################################


def calc_diameter(mesh):
    distance = []
    vertices = np.asarray(mesh.vertices)
    number = vertices.shape[0]
    comb = list(combinations(range(0, number), 2))
    for i in comb:
        distance.append(np.linalg.norm(vertices[i[0]] - vertices[i[1]]))
    return max(distance)


#open3d.geometry.sample_points_uniformly(samp)
#####################################################
###Calculate the eccentricity of the meshes
###Dividing the longest with the shortest eigen value
#####################################################

def calc_eccentricity(mesh):
    eigen_values, _ = calc_eigen(mesh)
    position = np.argsort(eigen_values)[::-1]
    x = eigen_values[position[0]]
    z = eigen_values[position[2]]
    eccentricity = x/z
    return eccentricity


#print(calc_eccentricity(mesh))

##################################
###Shape property distributions###
##################################

def vertex_output(mesh):
    vertex = np.asarray(mesh.vertices)
    num_vertex = vertex.shape[0]
    return vertex, num_vertex


# A3:
def A3(mesh, seed):
    vertex, num_vertex = vertex_output(mesh)
    theta = []
    random.seed(seed)
    for i in range(0, 50000):
        sample = random.sample(range(0, num_vertex), 3)
        v1 = vertex[sample[1]] - vertex[sample[0]]
        v2 = vertex[sample[2]] - vertex[sample[0]]
        #print('1st vertex: ', vertex[sample[1]])
        #print('2nd vertex: ', vertex[sample[2]])
        theta.append(degrees(acos(np.dot(v1, v2) / (np.linalg.norm(v2) * np.linalg.norm(v1)))))
    return theta
# draw the density plot
# sns.kdeplot(theta)


# D1
def D1(mesh, seed):
    vertex, _ = vertex_output(mesh)
    dist_bary = []
    random.seed(seed)
    indices = range(0, vertex.shape[0])
    for i in indices:
        v = vertex[i]
        dist_bary.append((np.linalg.norm(v)))
    return dist_bary
# draw density plot
# sns.kdeplot(dist_bary)


# D2
def D2(mesh, seed):
    vertex, num_vertex = vertex_output(mesh)
    dist_2 = []
    random.seed(seed)
    for i in range(0, 50000):
        sample = random.sample(range(0, num_vertex), 2)
        v_2 = vertex[sample[1]] - vertex[sample[0]]
        dist_2.append((np.linalg.norm(v_2)))
    return dist_2
# draw density plot
# sns.kdeplot(dist_2)


# D3
def D3(mesh, seed):
    vertex, num_vertex = vertex_output(mesh)
    area = []
    random.seed(seed)
    for i in range(0, 250000):
        sample = random.sample(range(0, num_vertex), 3)
        v1 = vertex[sample[1]] - vertex[sample[0]]
        v2 = vertex[sample[2]] - vertex[sample[0]]
        area.append(sqrt(0.5 * np.linalg.norm(np.cross(v1, v2))))
    return area
# sns.kdeplot(area)


# D4
def D4(mesh, seed):
    vertex, num_vertex = vertex_output(mesh)
    volume_D4 = []
    random.seed(seed)
    for i in range(0, 2000000):
        sample = random.sample(range(0, num_vertex), 4)
        v1 = vertex[sample[1]] - vertex[sample[0]]
        v2 = vertex[sample[2]] - vertex[sample[0]]
        v3 = vertex[sample[3]] - vertex[sample[0]]
        volume_D4.append((1/6 * abs(np.dot(np.cross(v1, v2), v3))) ** (1/3))
    return volume_D4
#sns.kdeplot(volume_D4)


###################################################
###Computing volume of a 3D mesh
###First, compute the single volume of the triangle
###Then, compute the whole mesh
###################################################

def diameter(mesh):
    distance = []
    vertices = np.asarray(mesh.vertices)
    number = vertices.shape[0]
    comb = list(combinations(range(0,number), 2))
    for i in comb:
        distance.append(np.linalg.norm(vertices[i[0]]-vertices[i[1]]))
    return max(distance)


#####################################################################
###calculating the compactness of the mesh
###The equation of compactness is C = S^3/ (36PI * V^2)
###To Sphericity: 1/C
#####################################################################



def calc_convex_volume(path):
    mesh = trimesh.load(path)
    volume = mesh.convex_hull.volume
    return volume


def calc_convex_surface(path):
    mesh = trimesh.load(path)
    con = mesh.convex_hull
    area = con.area
    return area


def calc_sphericity(path):
    volume = calc_convex_volume(path)
    area = calc_convex_surface(path)
    #print('area: ', area)
    #print('volume: ', volume)
    sphericity = 1/(area**3 / (36 * pi * (volume ** 2)))
    return sphericity


#print('sphericity: ', calc_sphericity(path))
#mesh = trimesh.load(path)
#mesh.show()




