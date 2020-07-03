def multilabel_fbeta_batch(output, target, beta=2, thresh=0.3, sigmoid=True):
    pred, targ = ((output.sigmoid() if sigmoid else output) > thresh).byte(), target.byte()
    m = pred * targ

    return m.sum(0).float(), pred.sum(0).float(), targ.sum(0).float()


def fbeta_score(precision, recall, beta=2, eps=1e-15):
    beta2 = beta ** 2
    return (1 + beta2) * (precision * recall) / ((beta2 * precision + recall) + eps)


def multilabel_fbeta_epoch(tp, pred, targ):
    precision, recall = tp.sum() / pred.sum(), tp.sum() / targ.sum()
    return fbeta_score(precision, recall)
