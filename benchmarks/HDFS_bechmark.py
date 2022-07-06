#!/usr/bin/env python
# -*- coding: utf-8 -*-
print("Called script")

import sys
import time
sys.path.append('../')
from loglizer import dataloader, preprocessing
from loglizer.models import LogClustering, InvariantsMiner, PCA, DeepLog
import pandas as pd
import json

run_models = ["PCA"]
hparams_search = True
print("Starting benchmark")

if __name__ == '__main__':
    x_tr, (x_te, y_test), (x_va, y_val) = dataloader.load_linux(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
                                                                window='session',
                                                                train_ratio=0.5,
                                                                split_type='uniform')

    benchmark_results = []
    feature_extractor = preprocessing.FeatureExtractor()

    for _model in run_models:
        print('Evaluating {} on Linux Logs:'.format(_model))
        if _model == 'PCA':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf',
                                                      normalization='zero-mean',num_keys=415)
            x_test = feature_extractor.transform(x_te)
            x_val = feature_extractor.transform(x_va)
            model = PCA(n_components=3, c_alpha=3.8906)
            t_s = time.time()
            model.fit(x_train)
            t_e = time.time()
            if hparams_search:
                for n_components in range(19):
                    for c_alpha in [1.7507, 1.9600, 2.5758, 2.807, 2.9677, 3.2905, 3.4808, 3.8906, 4.4172]:
                        model = PCA(n_components=n_components, c_alpha=c_alpha)
                        model.fit(x_train)
                        precision, recall, f1 = model.evaluate(x_test, y_test)
                        benchmark_results.append(
                            [_model + '-test-' + str(n_components) + "-" + str(c_alpha), precision, recall, f1])

        elif _model == 'InvariantsMiner':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, num_keys=415)
            x_test = feature_extractor.transform(x_te)
            x_val = feature_extractor.transform(x_va)
            t_s = time.time()
            model = InvariantsMiner(epsilon=0.1, percentage=1.0)
            model.fit(x_train)
            t_e = time.time()
            print(f"Time for Training: {t_e - t_s:.3f}s")
            pvals = [0.96,0.96,0.97,0.98,0.99,0.995,0.999,1.0]
            evals = [0.1]#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            i = 0
            if hparams_search:
                for p in pvals:
                    for e in evals:
                        i += 1
                        print(f"Testing combination {i}/{len(pvals) * len(evals)}")
                        model = InvariantsMiner(percentage=p, epsilon=e)
                        t_s = time.time()
                        model.fit(x_train)
                        t_e =time.time()
                        precision, recall, f1 = model.evaluate(x_test, y_test)
                        t_e2 = time.time()
                        benchmark_results.append(
                            [_model + '-test-' + str(p) + "-" + str(e), precision, recall, f1, t_e - t_s, t_e2 - t_e])


        elif _model == 'LogCluster':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(
                x_tr, term_weighting='tf-idf', num_keys=415)
            x_test = feature_extractor.transform(x_te)
            x_val = feature_extractor.transform(x_va)
            model = LogClustering(max_dist=0.1, anomaly_threshold=0.4)
            t_s = time.time()
            model.fit(x_train)  # Use only normal samples for training
            t_e = time.time()
            if hparams_search:
                for mdist in [0.1, 0.2, 0.3, 0.4]:
                    for at in [0.1, 0.2, 0.3, 0.4]:
                        model = LogClustering(
                            max_dist=mdist, anomaly_threshold=at)
                        model.fit(x_train)
                        precision, recall, f1 = model.evaluate(x_test, y_test)
                        benchmark_results.append(
                            [_model + '-test-' + str(mdist) + "-" + str(at), precision, recall, f1])
        elif _model == 'DeepLog':
            feature_extractor = preprocessing.Vectorizer()
            x_train = feature_extractor.fit_transform(
                x_tr,  num_keys=415)
            x_test = feature_extractor.transform(x_te)
            x_val = feature_extractor.transform(x_va)
            model = DeepLog(num_labels=415, device=-1)
            t_s = time.time()
            model.fit(x_train)  # Use only normal samples for training
            t_e = time.time()
            if hparams_search:
                for mdist in [0.1, 0.2, 0.3, 0.4]:
                    for at in [0.1, 0.2, 0.3, 0.4]:
                        model = LogClustering(
                            max_dist=mdist, anomaly_threshold=at)
                        model.fit(x_train)
                        precision, recall, f1 = model.evaluate(x_test, y_test)
                        benchmark_results.append(
                            [_model + '-test-' + str(mdist) + "-" + str(at), precision, recall, f1])

        print('Validation accuracy:')
        t_s_p = time.time()
        precision, recall, f1, y_pred = model.evaluate(x_val, y_val)
        t_e_p = time.time()
        with open(f"vecs-{_model}", "w") as f:
            f.write(json.dumps({"y_pred": y_pred.tolist(), "y_true": y_val.tolist()}))
        print(f"Time for Predicting: {t_e_p - t_s_p:.3f}s")
        benchmark_results.append([_model + '-val', precision, recall, f1, t_e -t_s, t_e_p - t_s_p])

    pd.DataFrame(benchmark_results, columns=['Model', 'Precision', 'Recall', 'F1', 't_train', 't_predict']) \
      .to_csv(f"benchmark_result_{_model}.csv", index=False)
