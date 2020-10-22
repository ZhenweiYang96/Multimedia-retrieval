from ui_function import *
from processing_distribution_function import *
import itertools
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support as score, \
    plot_roc_curve
from evaluation_function import plotting_roc_curve


def run():

    running = input('do you want to run all the shapes? y/n ')
    if running == 'y':
        mesh_df = pd.read_csv('excel_file\\matching_features.csv')
        mesh_df[['A3', 'D1', 'D2', 'D3', 'D4']] = mesh_df[['A3', 'D1', 'D2', 'D3', 'D4']].apply(
            feature_comma_adjustment)
        del mesh_df['Unnamed: 0']
        del mesh_df['Unnamed: 0.1']
    else:
        from_cache = input('load from cash? y/n ')
        if from_cache == 'n':

            file_path = input('Hellllllllllllo, input the mesh: ')
            # visual_data(file_path)

            ###Basic information of the mess
            # df = one_data_information(file_path)
            ###Process the mesh

            print('calculating features...')
            mesh = preprocess_mesh(file_path)
            mesh_df = feature_extraction(mesh, file_path)
            mesh_df = to_bin(mesh_df)

            print('Normalizing the features...')
            mean_std_data = pd.read_csv('excel_file\\standardized.csv')
            means = mean_std_data[['surface_area', 'sphericity', 'bounding_box_volume', 'diameter', 'eccentricity']].mean()
            std = mean_std_data[['surface_area', 'sphericity', 'bounding_box_volume', 'diameter', 'eccentricity']].std()
            mesh_df[['surface_area']] = single_value_normalize(mesh_df[['surface_area']], means[0], std[0])
            mesh_df[['sphericity']] = single_value_normalize(mesh_df[['sphericity']], means[1], std[1])
            mesh_df[['bounding_box_volume']] = single_value_normalize(mesh_df[['bounding_box_volume']], means[2], std[2])
            mesh_df[['diameter']] = single_value_normalize(mesh_df[['diameter']], means[3], std[3])
            mesh_df[['eccentricity']] = single_value_normalize(mesh_df[['eccentricity']], means[4], std[4])

            mesh_df[['A3', 'D1', 'D2', 'D3', 'D4']] = mesh_df[['A3', 'D1', 'D2', 'D3', 'D4']].apply(to_percentage)
            mesh_df[['A3', 'D1', 'D2', 'D3', 'D4']] = mesh_df[['A3', 'D1', 'D2', 'D3', 'D4']].apply(
                convert_scientific_notation)
            mesh_df.to_csv('processing\\mesh_feature.csv')
            mesh_df = pd.read_csv('processing\\mesh_feature.csv')
            mesh_df[['A3', 'D1', 'D2', 'D3', 'D4']] = mesh_df[['A3', 'D1', 'D2', 'D3', 'D4']].apply(
                feature_comma_adjustment)
            del mesh_df['Unnamed: 0']
        else:
            mesh_df = pd.read_csv('processing\\mesh_feature.csv')
            mesh_df[['A3', 'D1', 'D2', 'D3', 'D4']] = mesh_df[['A3', 'D1', 'D2', 'D3', 'D4']].apply(
                feature_comma_adjustment)
            del mesh_df['Unnamed: 0']

    ###All the numerical features
    distribution = mesh_df.iloc[:, 2:len(mesh_df.columns)]

    ###Calculate weights and adjust the values
    # weights = get_weights('excel_file\\matching_features.csv')
    has_weight = input('adjust weight? y/n ')
    if has_weight == 'y':
        print('adjusting the weights...')
        weights = pd.read_csv('excel_file\\weights.csv')
        weights = weights.loc[:, 'weights']
        weights = np.asarray(weights)
        weights = 1 / np.asarray(weights)
    else:
        weights = [1] * 10
    distribution.iloc[:, 0:5] = np.multiply(distribution.iloc[:, 0:5], weights[0:5])
    distribution['A3'] = [[i * weights[5] for i in row] for row in distribution['A3']]
    distribution['D1'] = [[i * weights[6] for i in row] for row in distribution['D1']]
    distribution['D2'] = [[i * weights[7] for i in row] for row in distribution['D2']]
    distribution['D3'] = [[i * weights[8] for i in row] for row in distribution['D3']]
    distribution['D4'] = [[i * weights[9] for i in row] for row in distribution['D4']]

    db_eval = pd.DataFrame(columns=['shape', 'precision', 'recall', 'accuracy'])
    no_retrieve = int(input('how many shapes do you want to retrieve? '))
    print('matching the features...')
    ###iterate the rows:
    for row in range(0, len(distribution)):
        ###Create a flattened list so we can use the EMD
        row_value = list(itertools.chain.from_iterable(distribution.iloc[row:row+1].values.tolist()))
        dist_list = row_value[0:5]
        array_flattening = np.hstack(row_value[5:10])
        dist_list.extend(array_flattening)
        result = matching(dist_list, has_weight)
        match = result.iloc[0:no_retrieve, 0:3]
        print(mesh_df)

        y_true = np.repeat(np.asarray(mesh_df.loc[row, 'class']), len(np.asarray(match.iloc[:, 1])), axis=0)
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
        shape = mesh_df.loc[row, 'class']
        db_eval = db_eval.append(pd.DataFrame([[shape, precision, recall, report['accuracy']]]))
        if running == 'n':
            print('Precision : {}'.format(precision))
            print('Recall    : {}'.format(recall))
            print('Accuracy : {}'.format(report['accuracy']))


    db_eval.to_csv('processing\\all_db_match.csv')


run()
