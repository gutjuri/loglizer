from glob import glob
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
)
import pandas as pd

models = ["PCA", "LogCluster", "InvariantsMiner", "DeepLog"]
models  =["PCA", "LogCluster", "InvariantsMiner"]
l_tr = 27742
l_val = 60486


def make_cm(vector, res):
    fig = plt.figure()
    # fig.suptitle("haha")
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.5)
    ax = gs.subplots()
    i = 0
    for x, m in sorted(vector.items(), key=lambda y: res[y[0]]["F1"][0]):
        if x == "InvariantsMiner":
            x = "Invariants Mining"
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
    for x, m in sorted(res.items(), key=lambda y: res[y[0]]["F1"]):
        if x.startswith("InvariantsMiner"):
            x = x.replace("InvariantsMiner", "Invariants Mining")
        for a, b in [(1.7507, 0.08), (1.9600, 0.05), (2.5758, 0.01), (2.807, 0.005), (2.9677, 0.003), (3.2905, 0.001), (3.4808,0.0005), (3.8906,0.0001), (4.4172,0.00001)]:
            x = x.replace(str(a), str(b))
        print(f"{x:17}&${m['Precision']:.3f}$&${m['Recall']:.3f}$&${m['F1']:.3f}$&{'?SI{'}{1000*m['t_train']/l_tr:.3f}{'}{?milli?second}'}&{'?SI{'}{1000*m['t_predict']/(l_val):.3f}{'}{?milli?second}??'}".replace("?", "\\"))

def output_t2(vector, res):
    for x, m in sorted(vector.items(), key=lambda y: res[y[0]]["F1"]):
        if x.startswith("InvariantsMiner"):
            x = x.replace("InvariantsMiner", "Invariants Mining")
        for a, b in [(1.7507, 0.08), (1.9600, 0.05), (2.5758, 0.01), (2.807, 0.005), (2.9677, 0.003), (3.2905, 0.001), (3.4808,0.0005), (3.8906,0.0001), (4.4172,0.00001)]:
            x = x.replace(str(a), str(b))
        
        m = confusion_matrix(m["y_true"], m["y_pred"])
        print(f"{x:17}&${m[1][1]}$&${m[0][1]}$&${m[0][0]}$&${m[1][0]}$??".replace("?", "\\"))

vectors = {}
results = {}
for m in models:
    fname_vector = f"benchmarks/vecs-{m}-*"
    for fname in glob(fname_vector):
        with open(fname) as f:
            js = json.load(f)
        vectors[fname.replace("benchmarks/vecs-", "")] = js
    df = pd.read_csv(f"benchmarks/benchmark_result_{m}.csv")
    for _, x in df.iterrows():
        #print(x)
        results[x["Model"]] = x

res_dl_d = "/lustre/work/ws/ws1/ul_csu94-deeplog/results/*-*-*-*.csv"

for fname in glob(res_dl_d):
    with open(fname) as f:
        df = pd.read_csv(fname)
    bn = os.path.basename(fname)
    modelname = f"DeepLog-{bn[:-4]}"
    for _, x in df.iterrows():
        results[modelname] = x
    print(df)

#make_cm(vectors, results)
output_t1(results)
print()
output_t2(vectors, results)