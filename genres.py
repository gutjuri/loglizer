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
import pandas as pd


models = ["PCA", "LogClustering", "InvariantsMiner"]

l_tr = 27742
l_val = 60486

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

def output_t1(res):
    for x, m in res.items():
        print(f"{x:17}&${m['Precision']:.3f}$&${m['Recall']:.3f}$&${m['F1']:.3f}$&{'?SI{'}{1000*m['t_fit']/l_tr:.3f}{'}{?milli?second}'}&{'?SI{'}{1000*m['t_predict']/(l_val):.3f}{'}{?milli?second}??'}".replace("?", "\\"))

def output_t2(vector):
    for x, m in vector.items():
        m = confusion_matrix(m["y_true"], m["y_pred"])
        print(f"{x:17}&${m['cm'][1][1]}$&${m['cm'][0][1]}$&${m['cm'][0][0]}$&${m['cm'][1][0]}$??".replace("?", "\\"))

vectors = {}
results = {}
for m in models:
    fname_vector = f"benchmarks/vecs-{m}"
    with open(fname_vector) as f:
        js = json.load(f)
    vectors[m] = js
    results[m] = pd.read_csv(f"benchmarks/benchmark_result_{m}.csv")
make_cm(vectors)
output_t1(results)
print()
output_t2(vectors)