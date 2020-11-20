# library and set directory
import copy
from preprocess_function import *
from step3_function import *
import re
from statistics import stdev
from processing_distribution_function import *
import numpy as np
import open3d
from scipy import stats
import os

# Step 1 read data
def visual_data(path, x_trans=0, y_trans=0, z_trans=0):
    mesh = open3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    mesh_mv = copy.deepcopy(mesh).translate((x_trans, y_trans, z_trans), relative=False)
    mesh_mv.compute_vertex_normals()
    open3d.visualization.draw_geometries([mesh, mesh_mv], mesh_show_wireframe=True)

def adjust_string_to_float(value):
    col = value.str.strip('[]')
    col = col.str.split(' ')
    for i in range(0, len(col)):
        col[i] = map(str.strip, col[i])
        col[i] = [x for x in col[i] if x.isdigit()]
        col[i] = [float(j) for j in col[i]]
        #print(col[i])
    return col


###Normalize the columns so mean = 0 and std = 1
def standardize_single_value(col):
    return (col - col.mean()) / col.std()


###columns to percentage for the distributions
def to_percentage(col):
    col = col.apply(lambda x: np.array(x))
    return col / col.map(lambda x: x.sum())


def convert_scientific_notation(col):
    for i in range(0, len(col)):
        col[i] = ["{:.7f}".format(value) for value in col[i]]
    return col


def preprocess_mesh(path):
    ###Preprocess the mesh, achieving normalization
    mesh = open3d.io.read_triangle_mesh(path)
    mesh = mesh_sampling(mesh)
    open3d.io.write_triangle_mesh('processing\\mesh.off', mesh)
    mesh = trimesh.load_mesh('processing\\mesh.off')
    mesh = basic_normalization(mesh)
    mesh = align_axis(mesh)
    mesh = flipping(mesh, 'flip')
    mesh = scaling(mesh)
    mesh.export('processing\\mesh.off')
    mesh = open3d.io.read_triangle_mesh('processing\\mesh.off')
    return mesh


def feature_extraction(mesh, mesh_path):
    mesh_df = pd.DataFrame(
        columns=['mesh_id', 'class', 'surface_area', 'sphericity', 'bounding_box_volume', 'diameter', 'eccentricity',
                 'A3', 'D1', 'D2', 'D3', 'D4'])
    seed_matrix = np.arange(0, 1900).reshape(380, 5)
    counter = 0
    ###Class shape and id
    splitting = mesh_path.split("\\")
    class_shape = splitting[6]
    sep = '.'
    mesh_id = splitting[8].split(sep, 1)[0]
    convex_surface = calc_convex_surface(mesh_path)
    sphericity = calc_sphericity(mesh_path)
    bounding_box_volume = calc_bounding_box_volume(mesh)
    diameter = calc_diameter(mesh)
    eccentricity = calc_eccentricity(mesh)
    a3 = A3(mesh, seed=seed_matrix[counter, 0])
    d1 = D1(mesh, seed=seed_matrix[counter, 1])
    d2 = D2(mesh, seed=seed_matrix[counter, 2])
    d3 = D3(mesh, seed=seed_matrix[counter, 3])
    d4 = D4(mesh, seed=seed_matrix[counter, 4])
    counter += 1
    mesh_df = mesh_df.append(
        pd.DataFrame([[mesh_id, class_shape, convex_surface, sphericity, bounding_box_volume, diameter, eccentricity,
                       a3, d1, d2, d3, d4]],
                     columns=mesh_df.columns))
    return mesh_df


def single_value_normalize(col, mean, std):
    return (col - mean) / std


def flat_value(iterate_list):
    value_list = []
    for i in iterate_list:
        if isinstance(i, np.ndarray):
            value_list.extend(np.concatenate([iterate_list]).ravel().tolist())
        else:
            value_list.append(i)
    return value_list


###Earth mover distance:
def earth_mover_distance(a, b):
    return stats.wasserstein_distance(a, b)


def euclidean_distance(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b))


def feature_adjustment(value):
    col = value.str.strip('[]')
    col = col.str.split(' ')
    for i in range(0, len(col)):
        col[i] = map(str.strip, col[i])
        col[i] = col[i] = [re.sub('[^\d\.]', '', s) for s in col[i]]
        col[i] = [float(j) for j in col[i]]
    return col


def feature_comma_adjustment(value):
    col = value.str.strip('[]')
    col = col.str.split(',')
    for i in range(0, len(col)):
        col[i] = map(str.strip, col[i])
        col[i] = col[i] = [re.sub('[^\d\.]', '', s) for s in col[i]]
        col[i] = [float(j) for j in col[i]]
    return col


def get_weights(database_name):
    ###Features of all the meshes we have seen so far
    database_feature = pd.read_csv(database_name)
    database_feature[['A3', 'D1', 'D2', 'D3', 'D4']] = database_feature[['A3', 'D1', 'D2', 'D3', 'D4']].apply(feature_adjustment)
    database_feature = database_feature.iloc[:, 2:len(database_feature.columns)]

    weights = []
    for i in range(2, len(database_feature.columns)):
        feature_weight = []
        for j in range(0, len(database_feature) - 1):
            for k in range(j + 1, len(database_feature)):
                weight_distance = euclidean_distance(database_feature.iloc[j, i], database_feature.iloc[k, i])
                feature_weight.append(weight_distance)
        weights.append(stdev(feature_weight))
    df = pd.DataFrame(weights, columns=['weights'])
    df.to_csv('excel_file\\weights.csv')
    return weights


# Converting links to html tags
def path_to_image_html(path, distance_df):
    #splitting = path.split("/")
    #sep = '.'
    #mesh_id = splitting[len(splitting) - 1].split(sep, 1)[0]
    distance_mesh = distance_df['distance'][distance_df['image'] == path]
    distance_mesh = distance_mesh.to_string()
    splits = distance_mesh.split(' ')
    distance_mesh = splits[4]
    sep = '\n'
    distance_mesh = distance_mesh.split(sep, 1)[0]
    #print(distance_mesh)
    image_html = '<img src="' + path + '" width="200"><br>' + '<center>distance = {}'.format(distance_mesh)
    return image_html


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def matching(single_feature, no_retrieve, running):

    ###Features of all the meshes we have seen so far
    database_feature = pd.read_csv('excel_file\\matching_features.csv')
    database_feature[['A3', 'D1', 'D2', 'D3', 'D4']] = database_feature[['A3', 'D1', 'D2', 'D3', 'D4']].apply(feature_adjustment)
    database_feature = database_feature.iloc[:, 2:len(database_feature.columns)]

    mesh_distance = []
    euc_list = []
    emd_list = []
    ###Id and shape
    mesh_info = database_feature.iloc[:, 0:2]

    weights = pd.read_csv('excel_file\\weights.csv')
    weights = weights.loc[:, 'weights']
    weights = np.asarray(weights)
    weights = 1 / np.asarray(weights)
    #print(weights/sum(weights))
    weights = weights / sum(weights)
    #weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for _, row in database_feature.iloc[:, 2:len(database_feature.columns)].iterrows():

        ###Create a flattened list so we can use the EMD
        single_value_vector = row[0:5].tolist()
        distribution_vector = np.hstack(row[5:10]).tolist()
        #distribution_vector[:] = [x / 20 for x in distribution_vector]
        #single_feature[5:len(single_feature)][:] = [x / 20 for x in single_feature[5:len(single_feature)]]

        sum_value = 0
        for i in range(0, 5):
            euc_dist = euclidean_distance(single_feature[i], single_value_vector[i])
            euc_dist = (euc_dist * weights[i]) / 375
            euc_list.append(euc_dist)
            sum_value += euc_dist
        for j in range(0, 100, 20):
            emd = earth_mover_distance(single_feature[j+5:j+25], distribution_vector[j:j+20]) * weights[int(j/20)]
            emd_list.append(emd)
            sum_value += emd
        mesh_distance.append(sum_value)
    mesh_info.loc[:, 'distance'] = mesh_distance

    ###Get the pictures
    image_list = []
    im_directory = 'mesh_picture\\image'
    for file_name in os.listdir(im_directory):
        image_list.append(os.path.join(im_directory, file_name))
    image_list.sort(key=natural_keys)
    mesh_info = mesh_info.sort_values(by=['mesh_id'])
    mesh_info.loc[:, 'image'] = image_list

    ###Replace the index with the first image (which is the original query shape
    mesh_info.index = np.repeat([mesh_info.iloc[0, 3]], len(mesh_info), axis=0).tolist()
    mesh_info = mesh_info.sort_values(by=['distance'])

    if running == 'y':
        result = mesh_info.iloc[0:no_retrieve]
        return result,  mesh_info.iloc[0:no_retrieve, 2:4]

    else:
        result = mesh_info.iloc[0:no_retrieve, len(mesh_info.columns) - 1:len(mesh_info.columns)]
        result_shape = result.transpose()
        result_shape.index = np.repeat([result_shape.iloc[0, 0]], 1, axis=0).tolist()
        ###Remove first query because it is the same
        #result_shape = result_shape.iloc[0, 1:len(result_shape.columns)]
        return result_shape, mesh_info.iloc[0:no_retrieve, 2:4]#, euc_list, emd_list

