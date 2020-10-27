# library
#import pandas as pd
#from Step4_matching_normalize_features import *
from ui_function import *
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn import metrics
#from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import NearestNeighbors
from matching_function import *
from IPython.core.display import HTML

#

def split(df, colname):
    for i in range(0,20):
        col = []
        for j in range(0,len(df)):
            col.append(df.loc[j, colname][i]/20)
        df[(colname+"_"+str(i))] = col

# len(stand_feature.columns)
# predict
#list(stand_feature.columns)
#X = stand_feature.iloc[:,3:108]
#Y = stand_feature[['class']]
#actual = []
#for item in Y:
#    actual.append(str(item).strip("\'[]"))

#knn = KNeighborsClassifier(n_neighbors=9).fit(X, Y)
#y_hat = knn.predict(np.array(X.iloc[1,:]).reshape(1,-1))
# plot the confusion matrix
#plot_confusion_matrix(knn, X, Y)


######
#df = stand_feature
#mesh_id = 62
#no_mesh=8
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

    #df_back = df.drop(df.index[df['mesh_id'] == mesh_id].to_list()[0],axis=0)
    query_shape = df[df['mesh_id'] == mesh_id].iloc[:,3:108]
    X = df.iloc[:,3:108]
    knn = NearestNeighbors(n_neighbors=no_mesh, algorithm='kd_tree').fit(X)
    distances_knn, _ = knn.kneighbors(np.asarray(query_shape).reshape(1,-1))
    distances_knn = distances_knn[0].tolist()
    #indices_knn = indices_knn[0].tolist()  # the output are similar shapes
    # start with rnn
    rnn = NearestNeighbors(radius=max(distances_knn)*1.05)
    rnn.fit(X)
    distances_rnn, indices_rnn = rnn.radius_neighbors(np.asarray(query_shape).reshape(1,-1))
    distances_rnn = distances_rnn[0].tolist()
    indices_rnn = indices_rnn[0].tolist()
    output_meshid = df.iloc[indices_rnn,1].tolist()
    output_class = df.iloc[indices_rnn,2].tolist()
    data = {'mesh_id': output_meshid,
            'class': output_class,
            'distance': distances_rnn,
            'image': ["3D database/image/" + str(x) + ".png" for x in output_meshid],
            'new_index': np.repeat(df[df['mesh_id'] == mesh_id].iloc[:, 2], len(output_meshid))
    }
    result = pd.DataFrame(data, columns = ["mesh_id", "class","distance","image", "new_index"] )
    result = result.sort_values('distance')
    result.to_html('image.html', escape=False, formatters=dict(image=path_to_image_html))
    return result

#### following is the library recommended by the teacher: ANN
# I couldn't find where to use radius NN
#from annoy import AnnoyIndex
#t = AnnoyIndex(105, "euclidean")
#for i in range(380):
   # v = np.asarray(X.loc[i,:])
   # t.add_item(i, v)



#t.build(19)
#indices, distances = t.get_nns_by_item(0, 8, include_distances = True)
#max(distances)*1.15