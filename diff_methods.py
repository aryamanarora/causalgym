import torch
from collections import defaultdict

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


# mean diff
def mean_diff(activations, labels):
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


@ignore_warnings(category=ConvergenceWarning)
def kmeans_diff(activations, labels):
    # fit kmeans
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(activations)
    
    # make vector
    vecs = kmeans.cluster_centers_
    vec = torch.tensor(vecs[0] - vecs[1], dtype=torch.float32)
    return vec / torch.norm(vec), None


def pca_diff(activations, labels):
    # fit pca
    pca = PCA(n_components=1).fit(activations)

    # get first component
    vec = torch.tensor(pca.components_[0], dtype=torch.float32)
    return vec / torch.norm(vec), None


def probing_diff(activations, labels):
    assert len(set(labels)) == 2
    label_0 = list(set(labels))[0]

    # fit linear model
    weight = torch.nn.Parameter(torch.zeros_like(activations[0]))
    weight.requires_grad = True

    # train
    optimizer = torch.optim.Adam([weight], lr=1e-3)
    for epoch in range(1):
        for i, (activation, label) in enumerate(zip(activations, labels)):
            logit = torch.matmul(weight, activation)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.tensor(1.0 if label == label_0 else 0.0))
            loss.backward()
            optimizer.step()
    
        # print acc
        acc = 0
        for activation, label in zip(activations, labels):
            logit = torch.matmul(weight, activation)
            prob = torch.sigmoid(logit).item()
            if (prob > 0.5 and label == label_0) or (prob < 0.5 and label != label_0):
                acc += 1

    weight_vector = weight.detach() / torch.norm(weight.detach())
    weight_vector.requires_grad = False
    return weight_vector, acc / len(activations)


def probing_diff_sklearn(activations, labels):
    # fit lr
    lr = LogisticRegression(random_state=0, max_iter=1000, fit_intercept=False).fit(activations, labels)
    accuracy = lr.score(activations, labels)

    # extract weight
    vec = torch.tensor(lr.coef_[0], dtype=torch.float32)
    return vec / torch.norm(vec), accuracy


def lda_diff(activations, labels):
    # fit lda
    lda = LinearDiscriminantAnalysis(n_components=1).fit(activations, labels)
    accuracy = lda.score(activations, labels)

    # extract weight
    vec = torch.tensor(lda.coef_[0], dtype=torch.float32)
    return vec / torch.norm(vec), accuracy


def random_diff(activations, labels):
    vec = torch.randn_like(activations[0])
    return vec / torch.norm(vec), None


method_to_class_mapping = {
    "mean": mean_diff,
    "kmeans": kmeans_diff,
    "probe": probing_diff_sklearn,
    "pca": pca_diff,
    "lda": lda_diff,
    "random": random_diff,
}