# library
from ui_function import *
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv("excel_file\\matching_features.csv")
df[['A3', 'D1', 'D2', 'D3', 'D4']] = df[['A3', 'D1', 'D2', 'D3', 'D4']].apply(
    feature_adjustment)
del df['Unnamed: 0']

def split(df, colname):
    for i in range(0,20):
        col = []
        for j in range(0,len(df)):
            col.append(df.loc[j, colname][i]/20)
        df[(colname+"_"+str(i))] = col

split(df,"A3")
split(df,"D1")
split(df,"D2")
split(df,"D3")
split(df,"D4")
stand_feature = df.drop(['A3','D1','D2','D3','D4'], axis=1)

# predict
#X = stand_feature.iloc[:,3:108]
#Y = stand_feature[['class']]
#actual = []
#for item in Y:
#    actual.append(str(item).strip("\'[]"))

### Classification with KNN
#knn = KNeighborsClassifier(n_neighbors=19).fit(X, Y)
#plot_confusion_matrix(knn, X, Y)

# calculate the distance
#shape1 = stand_feature.loc[0,"surface_area":"D4_19"]
#shape2 = stand_feature.loc[1,"surface_area":"D4_19"]
#distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(shape1, shape2)]))

def scalability(mesh_id ,df, k):
    df_back = df.drop(df.index[df['mesh_id'] == mesh_id].to_list()[0],axis=0) # first, we exclude the query shape
    query_shape = df[df['mesh_id'] == mesh_id].iloc[:,3:107] # the row for query shape
    X = df_back.iloc[:,3:107] # predictor
    knn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(X) # knn
    distances_knn, indices_knn = knn.kneighbors(np.asarray(query_shape).reshape(1,-1))
    distances_knn = distances_knn[0].tolist()
    indices_knn = indices_knn[0].tolist()  # the output are similar shapes
    # start with rnn
    rnn = NearestNeighbors(radius=max(distances_knn)*1.02) # use the largest distance * 1.02 as the radius
    rnn.fit(X)
    distances_rnn, indices_rnn = rnn.radius_neighbors(np.asarray(query_shape).reshape(1,-1))
    distances_rnn = distances_rnn[0].tolist()
    indices_rnn = indices_rnn[0].tolist()
    output_meshid = df_back.iloc[indices_rnn,1].tolist() # output the similar mesh id
    return distances_rnn, output_meshid # output the distance and similar mesh id

scalability(61, df, 8) # test

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