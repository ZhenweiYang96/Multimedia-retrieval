# library
#import pandas as pd
#from Step4_matching_normalize_features import *
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn import metrics
#from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import NearestNeighbors
from matching_function import *
from IPython.core.display import HTML
import time

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
#mesh_id = mesh_id.strip(' ').split(',')
#mesh_id =  [int(x) for x in mesh_id]
def scalability(mesh_id, no_mesh, single_mesh=False):
    #t0 = time.time()
    df = pd.read_csv("excel_file/matching_features.csv")
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
    X = df.iloc[:, 3:108]
    no_retrieve = no_mesh + 1
    result_df = pd.DataFrame()
    #t1 = time.time()
    #print('scalability time: ', t1 - t0)
    for item in mesh_id:
        query_shape = df[df['mesh_id'] == item].iloc[:, 3:108]
        knn = NearestNeighbors(n_neighbors=no_retrieve, algorithm='kd_tree').fit(X)
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
                'image': [r"C:\Users\Admin\Documents\GitHub\Multimedia-retrieval\proj\mesh_picture\image\\" + str(x) + ".png" for x in output_meshid],
                'new_index': np.repeat(df[df['mesh_id'] == item].iloc[:, 2], len(output_meshid))
        }
        match = pd.DataFrame(data, columns = ["mesh_id", "class","distance","image", "new_index"] )
        match = match.sort_values('distance')
        distance_df = match.iloc[:, 2:4]
        mesh_path = match.iloc[:, len(match.columns) - 2:len(match.columns)-1]
        mesh_path = mesh_path.transpose().iloc[0,:].tolist()
        #mesh_path.index = np.repeat([mesh_path.iloc[0, 0]], 1, axis=0).tolist()
        result = [path_to_image_html(x, distance_df) for x in mesh_path]
        result = pd.DataFrame([result])
        #result.to_html('image.html', escape=False, formatters=dict(image=path_to_image_html))
        result_df = result_df.append(result)
    if single_mesh == True:
        return match
    ###Change column name
    else:
        column_name = []
        for i in range(0, len(result_df.columns)):
            i = str(i)
            column_name.append(i)

        result_df.columns = column_name
        index_name = []
        for j in range(0, len(result_df)):
            col = result_df.iloc[j, 0]
                # print(col)
            index_name.append(col)

        result_df.index = index_name
        del result_df['0']
            ###Write the result

            ###Rendering the images in the dataframe using the HTML method.
            # HTML(result_df.to_html(escape=False, formatters=dict(image=path_to_image_html)))
        result_df.to_html('image_sc.html', escape=False)
        #return result_html

