# library
import math
from ui_function import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

df = pd.read_csv('excel_file\\standardized.csv')
df[['A3', 'D1', 'D2', 'D3', 'D4']] = df[['A3', 'D1', 'D2', 'D3', 'D4']] .apply(adjust_string_to_float)

colname = "A3"
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
del df['A3','D1','D2','D3','D4']
stand_feature = df.drop(['A3','D1','D2','D3','D4'], axis=1)

# predict
list(stand_feature.columns)
X = stand_feature.iloc[:,3:108]
Y = stand_feature[['class']]
#actual = []
#for item in Y:
#    actual.append(str(item).strip("\'[]"))

knn = KNeighborsClassifier(n_neighbors=19).fit(X, Y)
y_hat = knn.predict(X)
metrics.plot_roc_curve(knn, X, Y)
plot_confusion_matrix(knn, X, Y)

# calculate the distance
shape1 = stand_feature.loc[0,"surface_area":"D4_19"]
shape2 = stand_feature.loc[1,"surface_area":"D4_19"]
distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(shape1, shape2)]))
