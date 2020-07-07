def multilabel_fbeta_batch(output, target, beta=2, thresh=0.5, sigmoid=True, dim=1):
    pred, targ = ((output.sigmoid() if sigmoid else output) > thresh).byte(), target.byte()
    m = pred * targ

    return m.sum(dim=dim).float(), pred.sum(dim=dim).float(), targ.sum(dim=dim).float()


def fbeta_score(precision, recall, beta=2, eps=1e-15):
    beta2 = beta ** 2
    return (1 + beta2) * (precision * recall) / ((beta2 * precision + recall) + eps)


def multilabel_fbeta_epoch(tp, pred, targ):
    precision, recall = tp / pred, tp / targ
    return fbeta_score(precision, recall)
