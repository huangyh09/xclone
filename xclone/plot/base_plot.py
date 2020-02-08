import numpy as np
import seaborn as sns

WZcolors = np.array(['#4796d7', '#f79e54', '#79a702', '#df5858', '#556cab', 
                     '#de7a1f', '#ffda5c', '#4b595c', '#6ab186', '#bddbcf', 
                     '#daad58', '#488a99', '#f79b78', '#ffba00'])

def annoHeatMap(X, anno, cax_pos=[.99, .2, .03, .45], cmap="GnBu", **kwargs):
    """Heatmap with annotations
    """
    idx = np.argsort(np.dot(X, 2**np.arange(X.shape[1])) + 
                     anno * 2**X.shape[1])

    g = sns.clustermap(X[idx], cmap=cmap, yticklabels=False,
                       col_cluster=False, row_cluster=False,
                       row_colors=WZcolors[anno][idx], **kwargs)
    
    for idx, val in enumerate(np.unique(anno)):
        g.ax_col_dendrogram.bar(0, 0, color=WZcolors[idx],
                                label=val, linewidth=0)
    
    g.ax_col_dendrogram.legend(loc="center", ncol=6, title="True clone")
    g.cax.set_position(cax_pos)

    return g
