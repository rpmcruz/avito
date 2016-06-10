# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def fn(filename, myreader, info, lines):
    # Este encoding que eu faço aqui é por causa duma limitação do sklearn.
    # Estou a codificar categories como 83 como [0,0,0,1,0]. Ou seja, cada
    # categoria passa a ser um binário. Ele só funciona assim. Isto não é uma
    # limitação das árvores de decisão em teoria, mas é uma limitação do
    # sklearn.
    # Há outro software que podemos eventualmente usar que não precisa disto...
    # O xgboost tb não suporta categóricas.

    from sklearn.preprocessing import OneHotEncoder
    # NOTE: all pairs belong to the same category: we only need to use one
    encoding = OneHotEncoder(dtype=int, sparse=False)
    categories = info.iloc[lines[:, 0]].as_matrix(['categoryID'])
    encoding.fit(categories)
    categories01 = encoding.transform(categories)

    df = pd.read_csv('../data/Category.csv', dtype=int, index_col=0)
    encoding = OneHotEncoder(dtype=int, sparse=False)
    parents = df.ix[categories[:, -1]].as_matrix(['parentCategoryID'])
    encoding.fit(parents)
    parents01 = encoding.transform(parents)

    from utils.categorias import categorias
    names = ['"' + categorias[i] + '"' for i in np.unique(categories)]
    names += ['"'categorias[i] + '"' for i in np.unique(parents)]
    return ([categories01, parents01], names)
