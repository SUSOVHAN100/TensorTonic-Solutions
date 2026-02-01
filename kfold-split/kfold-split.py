import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """
    # Step 1: create indices
    indices = np.arange(N, dtype=int)

    # Step 2: shuffle if required
    if shuffle:
        if rng is not None:
            indices = rng.permutation(indices)
        else:
            np.random.shuffle(indices)

    # Step 3: compute fold sizes
    base = N // k
    extra = N % k   # first 'extra' folds get one extra element

    folds = []
    start = 0

    for i in range(k):
        fold_size = base + (1 if i < extra else 0)
        end = start + fold_size

        val_idx = indices[start:end]
        train_idx = np.concatenate((indices[:start], indices[end:]))

        folds.append((train_idx, val_idx))
        start = end

    return folds
