import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
)
import pandas as pd


models = ["PCA", "LogCluster", "InvariantsMiner", "DeepLog"]

l_tr = 27742
l_val = 60486


def make_cm(vector, res):
    fig = plt.figure()
    # fig.suptitle("haha")
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.5)
    ax = gs.subplots()
    i = 0
    for x, m in sorted(vector.items(), key=lambda y: res[y[0]]["F1"]):
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
        if x == "InvariantsMiner":
            x = "Invariants Mining"
        print(f"{x:17}&${m['Precision'][0]:.3f}$&${m['Recall'][0]:.3f}$&${m['F1'][0]:.3f}$&{'?SI{'}{1000*m['t_train'][0]/l_tr:.3f}{'}{?milli?second}'}&{'?SI{'}{1000*m['t_predict'][0]/(l_val):.3f}{'}{?milli?second}??'}".replace("?", "\\"))

def output_t2(vector, res):
    for x, m in sorted(vector.items(), key=lambda y: res[y[0]]["F1"]):
        if x == "InvariantsMiner":
            x = "Invariants Mining"
        m = confusion_matrix(m["y_true"], m["y_pred"])
        print(f"{x:17}&${m[1][1]}$&${m[0][1]}$&${m[0][0]}$&${m[1][0]}$??".replace("?", "\\"))

vectors = {}
results = {}
for m in models:
    fname_vector = f"benchmarks/vecs-{m}"
    with open(fname_vector) as f:
        js = json.load(f)
    vectors[m] = js
    results[m] = pd.read_csv(f"benchmarks/benchmark_result_{m}.csv")
make_cm(vectors, results)
output_t1(results)
print()
output_t2(vectors, results)