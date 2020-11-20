from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support as score
import seaborn as sn
from Scalability import *

def plot_accuracy(df):
    sorted(list(set(np.asarray(df.iloc[:, 3]))))
    df.set_index('new_index')
    y_true = df['new_index']
    y_pred = df['class']
    cm_df = pd.DataFrame(
        confusion_matrix(y_true,
                         y_pred,
                         labels=sorted(list(set(np.asarray(df.iloc[:, 3])))),
                         normalize='true'),
        index=sorted(list(set(np.asarray(df.iloc[:, 3])))),
        columns=sorted(list(set(np.asarray(df.iloc[:, 3])))))
    cm_df = np.round(cm_df, decimals=2)
    sn.heatmap(cm_df, annot=True, annot_kws={"size": 10})
    plt.show()

def evaluate_scalability(num_mesh):
    mesh_df = pd.read_csv("excel_file/matching_features.csv")
    mesh_df[['A3', 'D1', 'D2', 'D3', 'D4']] = mesh_df[['A3', 'D1', 'D2', 'D3', 'D4']].apply(
        feature_adjustment)
    del mesh_df['Unnamed: 0']
    split(mesh_df, "A3")
    split(mesh_df, "D1")
    split(mesh_df, "D2")
    split(mesh_df, "D3")
    split(mesh_df, "D4")
    mesh_df = mesh_df.drop(['A3', 'D1', 'D2', 'D3', 'D4'], axis=1)

    db_eval = pd.DataFrame(columns=['mesh_id', 'class', 'distance', 'new_index'])
    ###iterate the rows:
    #counter = 0
    #row = 0
    #num_mesh = 6

    for row in range(0, len(mesh_df)):
       # counter += 1
       # print(counter)

        mesh_id = mesh_df.iloc[row,1]
        #mesh_class = mesh_df.iloc[row,2]
        match = scalability([mesh_id],num_mesh, single_mesh=True)
        y_true = np.repeat(np.asarray(mesh_df.iloc[row, 2]), len(np.asarray(match.iloc[:, 1])), axis=0)
        y_pred = np.asarray(match.iloc[:, 1])
        report = classification_report(y_true=y_true,
                                       y_pred=y_pred,
                                       target_names=list(set(np.asarray(match.iloc[:, 1]))),
                                       output_dict=True, zero_division=True)

        precision, recall, _, _ = score(y_true, y_pred, average='macro', zero_division=True)
        db_eval = db_eval.append(match)
    db_eval.to_csv('processing/all_db_scalability.csv')
    plot_accuracy(db_eval)
    print('Precision : {}'.format(precision))
    print('Recall    : {}'.format(recall))
    print('Accuracy : {}'.format(report['accuracy']))
