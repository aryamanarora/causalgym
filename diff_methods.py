from numpy import add
import torch
from collections import defaultdict

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils._testing import ignore_warnings


# mean diff
def mean_diff(activations, labels, eval_activations, eval_labels):
    means, counts = {}, defaultdict(int)

    # accumulate
    for activation, label in zip(activations, labels):
        if label not in means:
            means[label] = torch.zeros_like(activation)
        means[label] += activation
        counts[label] += 1

    # calc means
    for k in means:
        means[k] /= counts[k]
    
    # make vector
    vecs = list(means.values())
    vec = vecs[1] - vecs[0]
    return vec / torch.norm(vec), None


@ignore_warnings(category=Warning)
def kmeans_diff(activations, labels, eval_activations, eval_labels):
    # fit kmeans
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(activations)
    
    # make vector
    vecs = kmeans.cluster_centers_
    vec = torch.tensor(vecs[0] - vecs[1], dtype=torch.float32)
    return vec / torch.norm(vec), None


def pca_diff(n_components=1):
    def diff_func(activations, labels, eval_activations, eval_labels):
        # fit pca
        pca = PCA(n_components=n_components).fit(activations)
        explained_variance = sum(pca.explained_variance_ratio_)

        # average all components
        vec = torch.tensor(pca.components_.mean(axis=0), dtype=torch.float32)
        return vec / torch.norm(vec), explained_variance
    return diff_func


def probe_diff(fit_intercept=False, penalty='l2', solver="lbfgs", C=1.0) -> callable:
    @ignore_warnings(category=Warning)
    def diff_func(activations, labels, eval_activations, eval_labels):
        # fit lr
        lr = LogisticRegression(random_state=0, max_iter=1000, l1_ratio=0.5,
                                fit_intercept=fit_intercept, C=C,
                                penalty=penalty, solver=solver).fit(activations, labels)
        accuracy = lr.score(eval_activations, eval_labels)

        # extract weight
        vec = torch.tensor(lr.coef_[0], dtype=torch.float32)
        return vec / torch.norm(vec), accuracy
    return diff_func


def lda_diff(activations, labels, eval_activations, eval_labels):
    # fit lda
    lda = LinearDiscriminantAnalysis(n_components=1).fit(activations, labels)
    accuracy = lda.score(eval_activations, eval_labels)

    # extract weight
    vec = torch.tensor(lda.coef_[0], dtype=torch.float32)
    return vec / torch.norm(vec), accuracy


def random_diff(activations, labels, eval_activations, eval_labels):
    vec = torch.randn_like(activations[0])
    return vec / torch.norm(vec), None


method_mapping = {
    "mean": mean_diff,
    "kmeans": kmeans_diff,
    "probe": probe_diff(fit_intercept=False, penalty='l2', solver="lbfgs", C=1.0),
    "pca": pca_diff(n_components=1),
    "lda": lda_diff,
    "random": random_diff,
}

additional_method_mapping = {
    # various pca components (up to 5)
    "pca_2": pca_diff(n_components=2),
    "pca_3": pca_diff(n_components=3),
    "pca_4": pca_diff(n_components=4),
    "pca_5": pca_diff(n_components=5),

    # various linear probe types
    "probe_noreg_noint": probe_diff(fit_intercept=False, penalty=None, solver="saga", C=1.0),
    "probe_noreg_int": probe_diff(fit_intercept=True, penalty=None, solver="saga", C=1.0),

    "probe_l1_noint_1": probe_diff(fit_intercept=False, penalty='l1', solver="saga", C=1.0),
    "probe_l2_noint_1": probe_diff(fit_intercept=False, penalty='l2', solver="saga", C=1.0),
    "probe_elastic_noint_1": probe_diff(fit_intercept=False, penalty="elasticnet", solver="saga", C=1.0),
    "probe_l1_int_1": probe_diff(fit_intercept=True, penalty='l1', solver="saga", C=1.0),
    "probe_l2_int_1": probe_diff(fit_intercept=True, penalty='l2', solver="saga", C=1.0),
    "probe_elastic_int_1": probe_diff(fit_intercept=True, penalty="elasticnet", solver="saga", C=1.0),

    "probe_l1_noint_0.1": probe_diff(fit_intercept=False, penalty='l1', solver="saga", C=0.1),
    "probe_l2_noint_0.1": probe_diff(fit_intercept=False, penalty='l2', solver="saga", C=0.1),
    "probe_elastic_noint_0.1": probe_diff(fit_intercept=False, penalty="elasticnet", solver="saga", C=0.1),
    "probe_l1_int_0.1": probe_diff(fit_intercept=True, penalty='l1', solver="saga", C=0.1),
    "probe_l2_int_0.1": probe_diff(fit_intercept=True, penalty='l2', solver="saga", C=0.1),
    "probe_elastic_int_0.1": probe_diff(fit_intercept=True, penalty="elasticnet", solver="saga", C=0.1),
    
    "probe_l1_noint_0.001": probe_diff(fit_intercept=False, penalty='l1', solver="saga", C=0.001),
    "probe_l2_noint_0.001": probe_diff(fit_intercept=False, penalty='l2', solver="saga", C=0.001),
    "probe_elastic_noint_0.001": probe_diff(fit_intercept=False, penalty="elasticnet", solver="saga", C=0.001),
    "probe_l1_int_0.001": probe_diff(fit_intercept=True, penalty='l1', solver="saga", C=0.001),
    "probe_l2_int_0.001": probe_diff(fit_intercept=True, penalty='l2', solver="saga", C=0.001),
    "probe_elastic_int_0.001": probe_diff(fit_intercept=True, penalty="elasticnet", solver="saga", C=0.001),

    "probe_l2_int_0.01": probe_diff(fit_intercept=True, penalty='l2', solver="saga", C=0.01),
    "probe_l2_int_0.0001": probe_diff(fit_intercept=True, penalty='l2', solver="saga", C=0.0001),
}