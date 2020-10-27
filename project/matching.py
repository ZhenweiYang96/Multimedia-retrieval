from matching_function import *
from processing_distribution_function import *
import itertools
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support as score, \
    plot_roc_curve
from evaluation import plot_accuracy
from IPython.core.display import HTML


def match_mesh(running, no_mesh, single_mesh=None):

    #running = input('do you want to run all the shapes? y/n ')
    if running == 'y':
        mesh_df = pd.read_csv('excel_file\\matching_features.csv')
        mesh_df[['A3', 'D1', 'D2', 'D3', 'D4']] = mesh_df[['A3', 'D1', 'D2', 'D3', 'D4']].apply(
            feature_comma_adjustment)
        del mesh_df['Unnamed: 0']

    else:
        file_path = single_mesh
        # visual_data(file_path)
        ###Basic information of the mess
        # df = one_data_information(file_path)
        ###Process the mesh
        print('calculating features...')
        db_df = pd.read_csv('excel_file\\matching_features.csv')
        del db_df['Unnamed: 0']
        db_df[['A3', 'D1', 'D2', 'D3', 'D4']] = db_df[['A3', 'D1', 'D2', 'D3', 'D4']].apply(
            feature_adjustment)
        splitting = file_path.split("\\")
        sep = '.'
        mesh_id = splitting[8].split(sep, 1)[0]
        mesh_df = db_df[db_df['mesh_id'] == int(mesh_id)]
        # mesh_df[['A3', 'D1', 'D2', 'D3', 'D4']] = mesh_df[['A3', 'D1', 'D2', 'D3', 'D4']].apply(convert_scientific_notation)
        # mesh_df.to_csv('processing\\mesh_feature.csv')
        # mesh_df = pd.read_csv('processing\\mesh_feature.csv')
        # mesh_df[['A3', 'D1', 'D2', 'D3', 'D4']] = mesh_df[['A3', 'D1', 'D2', 'D3', 'D4']].apply(
        #     feature_comma_adjustment)
        # del mesh_df['Unnamed: 0']

    ###All the numerical features
    distribution = mesh_df.iloc[:, 3:len(mesh_df.columns)]

    ###Calculate weights and adjust the values
    # weights = get_weights('excel_file\\matching_features.csv')

    #print('adjusting the weights...')
    # weights = pd.read_csv('excel_file\\weights.csv')
    # weights = weights.loc[:, 'weights']
    # weights = np.asarray(weights)
    # weights = 1 / np.asarray(weights)

    # distribution.iloc[:, 0:5] = np.multiply(distribution.iloc[:, 0:5], weights[0:5])
    # distribution['A3'] = [[i * weights[5] for i in row] for row in distribution['A3']]
    # distribution['D1'] = [[i * weights[6] for i in row] for row in distribution['D1']]
    # distribution['D2'] = [[i * weights[7] for i in row] for row in distribution['D2']]
    # distribution['D3'] = [[i * weights[8] for i in row] for row in distribution['D3']]
    # distribution['D4'] = [[i * weights[9] for i in row] for row in distribution['D4']]

    db_eval = pd.DataFrame(columns=['mesh_id', 'class', 'distance', 'new_index'])
    no_retrieve = no_mesh
    ###iterate the rows:
    counter = 0
    for row in range(0, len(distribution)):
        print(counter)
        ###Create a flattened list so we can use the EMD
        row_value = list(itertools.chain.from_iterable(distribution.iloc[row:row+1].values.tolist()))
        dist_list = row_value[0:5]
        array_flattening = np.hstack(row_value[5:10])
        dist_list.extend(array_flattening)
        result = matching(dist_list)
        match = result.iloc[0:no_retrieve, 0:len(result.columns)]
        new_index = np.repeat(np.asarray(mesh_df.iloc[row, 2]), no_retrieve, axis=0).tolist()
        match.loc[:, 'new_index'] = new_index

        ###Rendering the dataframe as HTML table
        match.to_html(escape=False, formatters=dict(image=path_to_image_html))

        ###Rendering the images in the dataframe using the HTML method.
        HTML(match.to_html(escape=False, formatters=dict(image=path_to_image_html)))

        match.to_html('image.html', escape=False, formatters=dict(image=path_to_image_html))
        #print(image_df)
        #print('eucledian: ', l2)
        #print('emd: ', emd)
        y_true = np.repeat(np.asarray(mesh_df.iloc[row, 2]), len(np.asarray(match.iloc[:, 1])), axis=0)
        y_pred = np.asarray(match.iloc[:, 1])
        cm_df = pd.DataFrame(
            confusion_matrix(y_true,
                             y_pred,
                             labels=list(set(np.asarray(match.iloc[:, 1]))),
                             normalize='true'),
            index=list(set(np.asarray(match.iloc[:, 1]))),
            columns=list(set(np.asarray(match.iloc[:, 1]))))

        report = classification_report(y_true=y_true,
                                       y_pred=y_pred,
                                       target_names=list(set(np.asarray(match.iloc[:, 1]))),
                                       output_dict=True)

        precision, recall, _, _ = score(y_true, y_pred, average='macro')

        db_eval = db_eval.append(match)
        if running == 'n':
            print('Precision : {}'.format(precision))
            print('Recall    : {}'.format(recall))
            print('Accuracy : {}'.format(report['accuracy']))
        counter += 1

    if running == 'y':
        db_eval.to_csv('processing\\all_db_match.csv')
        plot_accuracy(db_eval)


#match_mesh('n', 20)
