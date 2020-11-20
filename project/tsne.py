from Scalability import *
from sklearn.manifold import TSNE
import itertools


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

#df.iloc[:,3:8] = df.iloc[:,3:8]/400

X = df.iloc[:,3:108]

tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000,learning_rate=300)
tsne_result = tsne.fit_transform(X)
df['tsne-2d-one'] = tsne_result[:,0]
df['tsne-2d-two'] = tsne_result[:,1]

color_get = ["#FE0404", "#FD6503", "#F9FD03", "#CEFF33", "#58FF33",
         "#33FFDD", "#33CAFF", "#3380FF", "#3336FF", "#8333FF",
         "#541A7C", "#FF33FF", "#CF7BAA", "#080717", "#BABAC3",
         "#573A37", "#2F5C23", "#716F2F", "#2F3B71"]
    #"#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
plt.figure(figsize=(10,10))
#plt.title("tsne_p" + str(item[0]) + "_i" + str(item[1]) + "_l" +str(item[2]))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="class",
    #palette=sns.color_palette("tab10", 19),
    palette = sns.color_palette(color_get),
    data=df,
        #legend="full",
    alpha=0.9
)


perp = list(range(5,55,5))
iter  = [1000,5000]
learning = list(range(100, 700, 200))
grid = itertools.product(perp, iter, learning)

for item in grid:
    tsne = TSNE(n_components=2, verbose=1, perplexity=item[0], n_iter=item[1],learning_rate=item[2])
    tsne_result = tsne.fit_transform(X)
    df['tsne-2d-one'] = tsne_result[:,0]
    df['tsne-2d-two'] = tsne_result[:,1]

    plt.figure(figsize=(14,10))
    plt.title("tsne_p" + str(item[0]) + "_i" + str(item[1]) + "_l" +str(item[2]))
    figure = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="class",
        palette=sns.color_palette(color_get),
        data=df,
        #legend="full",
        alpha=0.9
    )
    figure = figure.get_figure()
    figure.savefig("processing/tsne_p" + str(item[0]) + "_i" + str(item[1]) + "_l" +str(item[2]))