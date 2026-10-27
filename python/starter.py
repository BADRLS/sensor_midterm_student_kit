"""Sensor analytics midterm - STUDENT STARTER (Python 3, standard library only).

Run from the kit root:      python3 python/starter.py
Run the supplied checks:    python3 python/check.py

Complete every function marked TODO. Do not change function names or signatures:
python/check.py imports them. Loading, partition boundaries, the k-NN accuracy
helper and all output formatting are supplied. IDs and fault labels must never
enter distance calculations or PCA.
"""
import csv
import heapq
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = ["temperature", "vibration", "pressure", "rpm"]
PARTITION_SIZES = [100, 300, 500, 700]        # sums to 1,600 training rows, in file order
K = 5
WORKED_EXAMPLE_ID = "R1377"                    # test-set reading used in Part 4


# ----------------------------------------------------------------------------- supplied: loading
def load_readings(path):
    """Returns a list of dicts: {'id': str, 'x': [4 floats], 'fault': int or None}."""
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.append({"id": row["id"],
                        "x": [float(row[c]) for c in FEATURES],
                        "fault": int(row["fault"]) if "fault" in row else None})
    return out


def load_stats(path):
    means, stds = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            means.append(float(row["mean"]))
            stds.append(float(row["std"]))
    return means, stds


def load_pca(mean_path, proj_path):
    with open(mean_path, newline="") as f:
        mean = [float(r["mean"]) for r in csv.DictReader(f)]
    with open(proj_path, newline="") as f:
        proj = [[float(r["pc1"]), float(r["pc2"])] for r in csv.DictReader(f)]   # 4 x 2
    return mean, proj


def load_nn(path):
    """Returns dict with W1 (3x4), b1 (3), v (3), c (float), eta (float)."""
    blocks, name, rows = {}, None, []
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        name, nrows = parts[0], int(parts[1])
        if len(parts) == 3:                     # matrix: name rows cols
            blocks[name] = [[float(t) for t in lines[i + 1 + r].split()] for r in range(nrows)]
            i += 1 + nrows
        else:                                   # vector: name length
            blocks[name] = [float(t) for t in lines[i + 1].split()]
            i += 2
    return {"W1": blocks["W1"], "b1": blocks["b1"], "v": blocks["v"],
            "c": blocks["c"][0], "eta": blocks["eta"][0]}


# ----------------------------------------------------------------------------- Part 1
def partition(readings, sizes):
    parts, start = [], 0
    for s in sizes:
        parts.append(readings[start:start + s])
        start += s
    return parts


def local_summary(part):
    """TODO (Part 1). Return (sums, count) for one partition.
    sums is a list of 4 floats: the per-feature sum of r['x'] over the partition."""
    count = len(part)
    sums = [0.0] * 4
    for r in part:
        x = r["x"] if isinstance(r, dict) and "x" in r else r
        for j in range(4):
            sums[j] += float(x[j])
    return sums, count


def merge_summaries(summaries):
    """TODO (Part 1). Merge a list of (sums, count) pairs into the 4 global feature means."""
    total_count = sum(count for _, count in summaries)
    if total_count == 0:
        return [0.0] * 4
    global_sums = [0.0] * 4
    for sums, _ in summaries:
        for j in range(4):
            global_sums[j] += sums[j]
    return [global_sums[j] / total_count for j in range(4)]


def mean_of_local_means(summaries):
    """TODO (Part 1). Compute each partition's local means, then average them WITHOUT
    weighting by count. This is the incorrect method you will compare against."""
    if not summaries:
        return [0.0] * 4
    # Support if passed summaries [(sums, count), ...] or partitions [[r1, r2, ...], ...]
    if isinstance(summaries[0], (list, tuple)) and len(summaries[0]) == 2 and isinstance(summaries[0][1], (int, float)):
        summ_list = summaries
    else:
        summ_list = [local_summary(p) for p in summaries]

    num_parts = len(summ_list)
    if num_parts == 0:
        return [0.0] * 4
    unweighted_means = [0.0] * 4
    for sums, count in summ_list:
        for j in range(4):
            unweighted_means[j] += (sums[j] / count) if count > 0 else 0.0
    return [unweighted_means[j] / num_parts for j in range(4)]


# ----------------------------------------------------------------------------- Part 2
def standardize(x, means, stds):
    """TODO (Part 2). Return the z-scores of the 4 features of x using the supplied
    training means and standard deviations."""
    vec = x["x"] if isinstance(x, dict) and "x" in x else x
    return [(vec[j] - means[j]) / stds[j] for j in range(len(means))]


def squared_distance(a, b):
    """TODO (Part 2). Squared Euclidean distance between two equal-length vectors.
    Do not take a square root."""
    return sum((float(u) - float(v)) ** 2 for u, v in zip(a, b))


def topk_sort(query, vectors, ids=None, k=K):
    """TODO (Part 2). Score every vector, sort, return the best k as [(id, sq_distance), ...]
    in ascending order of distance. Break ties by the smaller row index."""
    if isinstance(ids, int):
        k = ids
        ids = None
    if isinstance(query, dict) and "x" in query:
        query = query["x"]
    if vectors and isinstance(vectors[0], dict):
        if ids is None:
            ids = [r["id"] for r in vectors]
        vectors = [r["x"] for r in vectors]
    elif ids is None:
        ids = [str(i) for i in range(len(vectors))]

    scored = []
    for i, v in enumerate(vectors):
        d = squared_distance(query, v)
        scored.append((d, i))

    scored.sort(key=lambda item: (item[0], item[1]))
    limit = min(k, len(scored))
    return [(ids[i], d) for d, i in scored[:limit]]


def topk_heap(query, vectors, ids=None, k=K):
    """TODO (Part 2). Same output as topk_sort, but keep only a size-k MAX-heap of the
    best candidates seen so far. Python's heapq is a min-heap: store (-distance, -index)
    so that heap[0] is the worst of the current best k. Use heapq.heappush and
    heapq.heapreplace. Return the k results sorted ascending, like topk_sort."""
    if isinstance(ids, int):
        k = ids
        ids = None
    if isinstance(query, dict) and "x" in query:
        query = query["x"]
    if vectors and isinstance(vectors[0], dict):
        if ids is None:
            ids = [r["id"] for r in vectors]
        vectors = [r["x"] for r in vectors]
    elif ids is None:
        ids = [str(i) for i in range(len(vectors))]

    if k <= 0 or not vectors:
        return []

    heap = []
    for i, v in enumerate(vectors):
        d = squared_distance(query, v)
        cand_key = (-d, -i)
        if len(heap) < k:
            heapq.heappush(heap, (-d, -i, i, d))
        elif cand_key > (heap[0][0], heap[0][1]):
            heapq.heapreplace(heap, (-d, -i, i, d))

    best = sorted(heap, key=lambda item: (item[3], item[2]))
    return [(ids[item[2]], item[3]) for item in best]


# ----------------------------------------------------------------------------- Part 3
def project(z, pca_mean, proj):
    """TODO (Part 3). Subtract the supplied PCA mean from the standardized vector z, then
    multiply by the 4 x 2 projection matrix proj (proj[j][c] = row j, component c).
    Return a list of 2 floats."""
    raise NotImplementedError("TODO: project")


def neighbor_preservation(original_ids, reduced_ids):
    """TODO (Part 3). neighbor preservation@k: the number of IDs that appear in BOTH
    top-k lists (an integer from 0 to k)."""
    raise NotImplementedError("TODO: neighbor_preservation")


# ----------------------------------------------------------------------------- Part 4
def relu(a):
    return a if a > 0 else 0.0


def sigmoid(a):
    return 1.0 / (1.0 + math.exp(-a))


def forward(z, nn):
    """TODO (Part 4). a_i = sum_j W1[i][j] z_j + b1[i];  h_i = relu(a_i);
    y_hat = sigmoid(sum_i v[i] h_i + c).  Return (a, h, y_hat)."""
    raise NotImplementedError("TODO: forward")


def output_weight_update(z, y, nn, j):
    """TODO (Part 4). For output weight v_j (0-based j) return (gradient, new_value):
    dL/dv_j = (y_hat - y) h_j   and   v_j_new = v_j - eta * dL/dv_j.
    Do not modify nn."""
    raise NotImplementedError("TODO: output_weight_update")


def bce(y_hat, y):
    return -(y * math.log(y_hat) + (1 - y) * math.log(1 - y_hat))


# ----------------------------------------------------------------------------- supplied: reporting
def knn_fault_accuracy(train_vecs, train_labels, test_vecs, test_labels, topk, ids):
    """Supplied helper: majority vote of the 5 nearest training readings."""
    label_of = dict(zip(ids, train_labels))
    correct = 0
    for v, y in zip(test_vecs, test_labels):
        votes = sum(label_of[i] for i, _ in topk(v, train_vecs, ids))
        correct += int((votes >= 3) == bool(y))
    return correct / len(test_vecs)


def main():
    d = lambda *p: os.path.join(ROOT, *p)
    train = load_readings(d("data", "train.csv"))
    test = load_readings(d("data", "test.csv"))
    queries = load_readings(d("data", "queries.csv"))
    means, stds = load_stats(d("params", "feature_stats.csv"))
    pca_mean, proj = load_pca(d("params", "pca_mean.csv"), d("params", "pca_projection.csv"))
    nn = load_nn(d("params", "nn_weights.txt"))
    out = []
    p = out.append

    def part1():
        p("## Part 1 - Partitioned analytics\n")
        summaries = [local_summary(part) for part in partition(train, PARTITION_SIZES)]
        p("| Partition | Count | " + " | ".join(f"Local mean {f}" for f in FEATURES) + " |")
        p("|---|---|" + "---|" * 4)
        for i, (s, n) in enumerate(summaries):
            p(f"| P{i+1} | {n} | " + " | ".join(f"{x/n:.4f}" for x in s) + " |")
        g, w = merge_summaries(summaries), mean_of_local_means(summaries)
        p("\n| Feature | Merged global mean | Mean of local means | Error |")
        p("|---|---|---|---|")
        for j, f in enumerate(FEATURES):
            p(f"| {f} | {g[j]:.4f} | {w[j]:.4f} | {w[j]-g[j]:+.4f} |")

    def parts23():
        ids = [r["id"] for r in train]
        Z = [standardize(r["x"], means, stds) for r in train]
        p("\n## Parts 2-3 - Exact similarity search and PCA\n")
        p("| Query | Original-space top-5 (ID: sq. dist) | PCA-space top-5 (ID: sq. dist) | Preservation@5 | Heap = sort? |")
        p("|---|---|---|---|---|")
        fmt = lambda res: ", ".join(f"{i}: {dd:.3f}" for i, dd in res)
        try:
            Y = [project(z, pca_mean, proj) for z in Z]
        except NotImplementedError:
            Y = None
        total = 0
        for q in queries:
            zq = standardize(q["x"], means, stds)
            o_sort, o_heap = topk_sort(zq, Z, ids), topk_heap(zq, Z, ids)
            if Y is None:
                p(f"| {q['id']} | {fmt(o_sort)} | (Part 3 not done) | - | {'yes' if o_sort == o_heap else 'NO'} |")
                continue
            yq = project(zq, pca_mean, proj)
            r_sort, r_heap = topk_sort(yq, Y, ids), topk_heap(yq, Y, ids)
            pres = neighbor_preservation([i for i, _ in o_sort], [i for i, _ in r_sort])
            total += pres
            p(f"| {q['id']} | {fmt(o_sort)} | {fmt(r_sort)} | {pres}/5 | "
              f"{'yes' if o_sort == o_heap and r_sort == r_heap else 'NO'} |")
        if Y is None:
            return
        p(f"\nMean neighbor preservation@5: {total/len(queries):.1f}/5 ({100*total/(5*len(queries)):.0f}%)")
        Zt = [standardize(r["x"], means, stds) for r in test]
        Yt = [project(z, pca_mean, proj) for z in Zt]
        lab, tl = [r["fault"] for r in train], [r["fault"] for r in test]
        base = 1 - sum(tl) / len(tl)
        p(f"\nTest-set 5-NN fault accuracy: original {knn_fault_accuracy(Z, lab, Zt, tl, topk_heap, ids):.3f}, "
          f"PCA {knn_fault_accuracy(Y, lab, Yt, tl, topk_heap, ids):.3f}, always-'no fault' baseline {base:.3f}")

    def part4():
        ex = next(r for r in test if r["id"] == WORKED_EXAMPLE_ID)
        z = standardize(ex["x"], means, stds)
        a, h, y_hat = forward(z, nn)
        p(f"\n## Part 4 - Worked update for {ex['id']} (fault = {ex['fault']})\n")
        p("z = [" + ", ".join(f"{x:.4f}" for x in z) + "]")
        p("a = [" + ", ".join(f"{x:.4f}" for x in a) + "]")
        p("h = [" + ", ".join(f"{x:.4f}" for x in h) + "]")
        s = sum(nn["v"][i] * h[i] for i in range(3)) + nn["c"]
        p(f"output pre-activation s = {s:.4f};  y_hat = {y_hat:.4f};  loss = {bce(y_hat, ex['fault']):.4f}")
        for j in range(3):
            grad, new = output_weight_update(z, ex["fault"], nn, j)
            p(f"v_{j+1}: dL/dv = ({y_hat:.4f} - {ex['fault']}) x {h[j]:.4f} = {grad:+.4f};  "
              f"v_new = {nn['v'][j]:.2f} - {nn['eta']:.2f} x {grad:+.4f} = {new:.4f}")
        Zt = [standardize(r["x"], means, stds) for r in test]
        tl = [r["fault"] for r in test]
        acc = sum(int((forward(zz, nn)[2] >= 0.5) == bool(y)) for zz, y in zip(Zt, tl)) / len(tl)
        p(f"\nTest-set accuracy of the supplied network (threshold 0.5): {acc:.3f}")

    for part in (part1, parts23, part4):
        try:
            part()
        except NotImplementedError as e:
            p(f"\n[{part.__name__} not finished: {e}]\n")

    text = "\n".join(out)
    print(text)
    with open(os.path.join(ROOT, "python", "results.md"), "w") as f:
        f.write("# Results\n\n" + text + "\n")


if __name__ == "__main__":
    main()
