import json
import sys
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from datetime import datetime
from collections import defaultdict


models = ["PCA", "LogCluster", "InvariantsMining", "DeepLog"]

def make_cm(res):
    fig = plt.figure()
    # fig.suptitle("haha")
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.5)
    ax = gs.subplots()
    i = 0
    for x, m in res.items():
        axis = ax[i % 2, i // 2]
        axis.tick_params(labelsize=8)
        axis.tick_params(axis="y", labelrotation=45)
        disp = ConfusionMatrixDisplay.from_predictions(
            m["y_true"],
            m["y_pred"],
            normalize="pred",
            ax=axis,
            display_labels=["Normal", "Abnormal"],
            colorbar=False,
            #cmap="plasma"
            values_format='.4f'
        )
        axis.set_title(x)
        disp.im_.set_clim(0, 1)
        # disp.plot(cmap="plasma", )
        # ax[0,0] = disp.figure_
        #
        # fig.plot()
        i += 1
    fig.colorbar(disp.im_, ax=ax)
    plt.savefig(f"/lustre/work/ws/ws1/ul_csu94-test/Graphics/ad-all.svg")


vectors = {}
for m in models:
    fname_vector = f"vecs-{m}"
    js = json.load(fname_vector)
    vectors[m] = js
make_cm(vectors)