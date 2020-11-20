from matching_function import *
from processing_distribution_function import *
import itertools
import time
from evaluation import plot_accuracy


def match_mesh(running, no_mesh, mesh_id=None):

    #running = input('do you want to run all the shapes? y/n ')
    if running == 'y':
        mesh_df = pd.read_csv('excel_file\\matching_features.csv')
        mesh_df[['A3', 'D1', 'D2', 'D3', 'D4']] = mesh_df[['A3', 'D1', 'D2', 'D3', 'D4']].apply(
            feature_comma_adjustment)
        del mesh_df['Unnamed: 0']

    else:
        ###Process the mesh
        print('calculating features...')
        db_df = pd.read_csv('excel_file\\matching_features.csv')
        del db_df['Unnamed: 0']
        db_df[['A3', 'D1', 'D2', 'D3', 'D4']] = db_df[['A3', 'D1', 'D2', 'D3', 'D4']].apply(
            feature_adjustment)
        mesh_id = map(int, mesh_id)

        mesh_df = db_df[db_df['mesh_id'].isin(mesh_id)]

    ###All the numerical features
    distribution = mesh_df.iloc[:, 3:len(mesh_df.columns)]


    db_eval = pd.DataFrame(columns=['mesh_id', 'class', 'distance', 'new_index'])
    no_retrieve = no_mesh + 1
    ###iterate the rows:
    counter = 0

    ###Dataframe for selecting few meshes, showing the query shapes
    result_df = pd.DataFrame()
    t0 = time.time()
    for row in range(0, len(distribution)):
        print(counter)
        ###Create a flattened list so we can use the EMD
        row_value = list(itertools.chain.from_iterable(distribution.iloc[row:row+1].values.tolist()))
        dist_list = row_value[0:5]
        array_flattening = np.hstack(row_value[5:10])
        dist_list.extend(array_flattening)
        result, distance_df = matching(dist_list, no_retrieve, running)
        if running == 'y':
            match = result.iloc[0:no_retrieve, 0:len(result.columns)]
            new_index = np.repeat(np.asarray(mesh_df.iloc[row, 2]), no_retrieve, axis=0).tolist()
            match.loc[:, 'new_index'] = new_index
            db_eval = db_eval.append(match)

        if running == 'n':
            result = result.applymap(lambda x: path_to_image_html(x, distance_df))
            result_df = result_df.append(result)

        counter += 1

    if running == 'y':
        t1 = time.time()
        print(t1 - t0)
        plot_accuracy(db_eval)

    if running == 'n':
        ###Rendering the dataframe as HTML table

        ###Change column name
        column_name = []
        for i in range(0, len(result_df.columns)):
            i = str(i)
            column_name.append(i)

        result_df.columns = column_name
        index_name = []
        for j in range(0, len(result_df)):
            col = result_df.iloc[j, 0]
            #print(col)
            index_name.append(col)

        result_df.index = index_name
        del result_df['0']
        ###Write the result

        ###Rendering the images in the dataframe using the HTML method.
        #HTML(result_df.to_html(escape=False, formatters=dict(image=path_to_image_html)))

        result_df.to_html('image.html', escape=False)
#match_mesh('n', 20)
