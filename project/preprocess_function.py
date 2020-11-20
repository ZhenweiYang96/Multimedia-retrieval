###library and set directory
import numpy as np
import trimesh


###Functions
###This one aint working yet

###Functions
def mesh_sampling(mesh):
    mesh = mesh.subdivide_loop(number_of_iterations=1)
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=4000)
    return mesh


def basic_normalization(mesh):
    # translation to the barycenter
    mesh = mesh.apply_translation(-mesh.centroid)
    return mesh

def scaling(mesh):
    ###get the x y z
    #extent_length = mesh_test.bounding_box.extents
    ###choose the longest side (x/y/z)
    #scale_value = max(extent_length[0], extent_length[1], extent_length[2])
    ###use the longest one to scale
    #mesh = mesh.scale(1 / scale_value, center=np.asarray([0,0,0]))
    S = trimesh.transformations.scale_and_translate(1 / max(mesh.bounding_box.extents))
    mesh.apply_transform(S)
    return mesh


def align_axis(mesh):
    # make a 3 * n matrix for the coordinates
    A = np.asmatrix(np.transpose(np.asarray(mesh.vertices)))

    # covariance matrix
    A_cov = np.cov(A)  # 3x3 matrix
    # eigenvectors and eigenvalues
    eigen_values, eigen_vectors = np.linalg.eig(A_cov)

    position = np.argsort(eigen_values)[::-1]  # align longest eigenvector with x axis
    x1 = eigen_vectors[:, position[0]]
    y1 = eigen_vectors[:, position[1]]
    z1 = np.cross(x1,y1)
    rotation_mat = np.linalg.inv(np.column_stack((x1, y1, z1)))
    R = np.zeros((4, 4))
    R[:3, :3] = rotation_mat
    #R[:3, 3] = -mesh_test.centroid
    R[3, 3] = 1
    ###Rotate the mesh
    mesh = mesh.apply_transform(R)
    return mesh

def flipping(mesh, usage):
    #triangle = np.asarray(mesh.triangles)
    #vertice = np.asarray(mesh.vertices)
    vertice = mesh.vertices
    triangle = mesh.faces
    ###Count for usage mesh:
    is_flipped = 0

    ###Get the center of each triangle
    ###Total sum
    fi = [0, 0, 0]
    for i in triangle:
        j = i
        coordinates = []
        ###Get coordinates of the points
        for j in i:
            # print("coordinates of the points: ", vertices[j])
            ###List of the coordinates of the three points
            coordinates.append(vertice[j])
        ###sum the coordinates to get the center point
        x_coord = sum(k[0] for k in coordinates) / 3
        y_coord = sum(k[1] for k in coordinates) / 3
        z_coord = sum(k[2] for k in coordinates) / 3
        ###Store coordinates in numpy
        center_coord = [x_coord, y_coord, z_coord]
        fi += np.sign(center_coord) * np.square(center_coord)
    ###Transformation matrix
    # print(fi[0], fi[1], fi[2])
    F = [[np.sign(fi[0]), 0, 0, 0], [0, np.sign(fi[1]), 0, 0],
         [0, 0, np.sign(fi[2]), 0],
         [0, 0, 0, 1]]

    if usage == 'flip':
        mesh = mesh.apply_transform(F)
        return mesh
    if usage == 'hist':
        #print(F)
        if np.where(F == 1):
            is_flipped += 1
            #print('Pos')
        elif np.where(F == -1):
            is_flipped = 0
            #print('Neg')
        return is_flipped

def calc_eigen(mesh):
    A = np.asmatrix(np.transpose(np.asarray(mesh.vertices)))
    # covariance matrix
    A_cov = np.cov(A)  # 3x3 matrix
    # eigenvectors and eigenvalues
    eigen_values, eigen_vectors = np.linalg.eig(A_cov)
    return eigen_values, eigen_vectors