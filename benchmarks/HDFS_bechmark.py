#!/usr/bin/env python
# -*- coding: utf-8 -*-
print("Called script")

import sys
import time
sys.path.append('../')
from loglizer import dataloader, preprocessing
from loglizer.models import LogClustering, InvariantsMiner, PCA
import pandas as pd

run_models = ["InvariantsMiner"]
hparams_search = False
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
            model = PCA(n_components=18)
            model.fit(x_train)
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
            model = InvariantsMiner(epsilon=0.5)
            model.fit(x_train)
            t_e = time.time()
            print(f"Time for Training: {t_e - t_s:.3f}s")
            if hparams_search:
                for p in [0.96,0.96,0.97,0.98,0.99,0.995,0.999,1.0]:
                    for e in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                        model = InvariantsMiner(percentage=p, epsilon=e)
                        model.fit(x_train)
                        precision, recall, f1 = model.evaluate(x_test, y_test)
                        benchmark_results.append(
                            [_model + '-test-' + str(p) + "-" + str(e), precision, recall, f1])


        elif _model == 'LogClustering':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(
                x_tr, term_weighting='tf-idf', num_keys=415)
            x_test = feature_extractor.transform(x_te)
            x_val = feature_extractor.transform(x_va)
            model = LogClustering(max_dist=0.3, anomaly_threshold=0.3)
            model.fit(x_train)  # Use only normal samples for training
            if hparams_search:
                for mdist in [0.1, 0.2, 0.3, 0.4]:
                    for at in [0.1, 0.2, 0.3, 0.4]:
                        model = LogClustering(
                            max_dist=mdist, anomaly_threshold=at)
                        model.fit(x_train)
                        precision, recall, f1 = model.evaluate(x_test, y_test)
                        benchmark_results.append(
                            [_model + '-test-' + str(mdist) + "-" + str(at), precision, recall, f1])

        x_test = feature_extractor.transform(x_te)
        x_val = feature_extractor.transform(x_va)
        print('Validation accuracy:')
        t_s = time.time()
        precision, recall, f1 = model.evaluate(x_val, y_val)
        t_e = time.time()
        print(f"Time for Predicting: {t_e - t_s:.3f}s")
        benchmark_results.append([_model + '-val', precision, recall, f1])

    pd.DataFrame(benchmark_results, columns=['Model', 'Precision', 'Recall', 'F1']) \
      .to_csv('benchmark_result.csv', index=False)
