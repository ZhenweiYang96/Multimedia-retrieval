# library
from ui_function import *
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import NearestNeighbors

def scalability(mesh_id, no_mesh):
    df = pd.read_csv("excel file/matching_features.csv")
    df[['A3', 'D1', 'D2', 'D3', 'D4']] = df[['A3', 'D1', 'D2', 'D3', 'D4']].apply(
        feature_adjustment)
    del df['Unnamed: 0']

    split(df, "A3")
    split(df, "D1")
    split(df, "D2")
    split(df, "D3")
    split(df, "D4")
    df = df.drop(['A3', 'D1', 'D2', 'D3', 'D4'], axis=1)

    df_back = df.drop(df.index[df['mesh_id'] == mesh_id].to_list()[0], axis=0)
    query_shape = df[df['mesh_id'] == mesh_id].iloc[:, 3:108]
    X = df_back.iloc[:,3:108]
    knn = NearestNeighbors(n_neighbors=no_mesh, algorithm='kd_tree').fit(X)
    distances_knn, _ = knn.kneighbors(np.asarray(query_shape).reshape(1, -1))
    distances_knn = distances_knn[0].tolist()
    #indices_knn = indices_knn[0].tolist()  # the output are similar shapes
    # start with rnn
    rnn = NearestNeighbors(radius=max(distances_knn)*1.05)
    rnn.fit(X)
    distances_rnn, indices_rnn = rnn.radius_neighbors(np.asarray(query_shape).reshape(1, -1))
    distances_rnn = distances_rnn[0].tolist()
    indices_rnn = indices_rnn[0].tolist()
    output_meshid = df_back.iloc[indices_rnn, 1].tolist()
    output_class = df_back.iloc[indices_rnn, 2].tolist()
    data = {'mesh_id': output_meshid,
            'class': output_class,
            'distance': distances_rnn,
            'new_index': np.repeat(df[df['mesh_id'] == mesh_id].iloc[:, 2], len(output_meshid)),
            'image': ["3D database\\image\\" + str(x) + ".png" for x in output_meshid]
    }
    result = pd.DataFrame(data, columns = ["mesh_id", "class","distance","new_index", "image"] )
    return result
