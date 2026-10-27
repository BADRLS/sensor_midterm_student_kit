"""Supplied checks (Part 6). Run from the kit root:

    python3 python/check.py                 # checks python/starter.py
    python3 python/check.py path/to/file.py # checks another file with the same functions

The checks use small fixtures and invariants. Passing them shows your functions behave
correctly; it does not reveal the exam answers.
"""
import copy
import importlib.util
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "python", "starter.py")
spec = importlib.util.spec_from_file_location("student", target)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

results = []


def check(name, fn):
    try:
        fn()
        results.append((name, "PASS", ""))
    except NotImplementedError as e:
        results.append((name, "TODO", str(e)))
    except AssertionError as e:
        results.append((name, "FAIL", str(e)))
    except Exception as e:                                   # noqa: BLE001
        results.append((name, "ERROR", f"{type(e).__name__}: {e}"))


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def R(i, x, fault=0):
    return {"id": f"T{i}", "x": list(x), "fault": fault}


train = S.load_readings(os.path.join(ROOT, "data", "train.csv"))
means, stds = S.load_stats(os.path.join(ROOT, "params", "feature_stats.csv"))
pca_mean, proj = S.load_pca(os.path.join(ROOT, "params", "pca_mean.csv"),
                            os.path.join(ROOT, "params", "pca_projection.csv"))
nn = S.load_nn(os.path.join(ROOT, "params", "nn_weights.txt"))
rng = random.Random(11)


# ---------------------------------------------------------------- Part 1
def t_local_summary():
    sums, n = S.local_summary([R(1, [1, 2, 3, 4], 1), R(2, [5, 6, 7, 8], 0)])
    assert n == 2, f"count should be 2, got {n}"
    assert all(close(a, b) for a, b in zip(sums, [6, 8, 10, 12])), f"sums wrong: {sums}"


def t_merge_fixture():
    parts = [[R(1, [0, 0, 0, 0])], [R(2, [10, 10, 10, 10]), R(3, [10, 10, 10, 10]), R(4, [10, 10, 10, 10])]]
    summ = [S.local_summary(p) for p in parts]
    g = S.merge_summaries(summ)
    assert all(close(x, 7.5) for x in g), f"merged mean should be 7.5, got {g}"
    w = S.mean_of_local_means(summ)
    assert all(close(x, 5.0) for x in w), f"mean of local means should be 5.0 here, got {w}"


def t_merge_real():
    summ = [S.local_summary(p) for p in S.partition(train, S.PARTITION_SIZES)]
    g = S.merge_summaries(summ)
    direct = [sum(r["x"][j] for r in train) / len(train) for j in range(4)]
    assert all(close(a, b, 1e-9) for a, b in zip(g, direct)), "merged means differ from a single-pass mean"
    assert sum(n for _, n in summ) == len(train) == 1600, "partition counts must total 1,600"


# ---------------------------------------------------------------- Part 2
def t_standardize():
    z = S.standardize(means, means, stds)
    assert all(close(x, 0) for x in z), "the training mean should standardize to 0"
    x = [means[j] + 2 * stds[j] for j in range(4)]
    assert all(close(v, 2) for v in S.standardize(x, means, stds)), "mean + 2 sd should give 2"
    Z = [S.standardize(r["x"], means, stds) for r in train]
    for j in range(4):
        m = sum(z[j] for z in Z) / len(Z)
        assert abs(m) < 1e-3, f"standardized training feature {j} has mean {m:.4f}, expected ~0"


def t_sqdist():
    assert close(S.squared_distance([0, 0], [3, 4]), 25), "d^2((0,0),(3,4)) should be 25 (no square root)"
    assert close(S.squared_distance([1, 2, 3, 4], [1, 2, 3, 4]), 0)
    a, b = [rng.uniform(-3, 3) for _ in range(4)], [rng.uniform(-3, 3) for _ in range(4)]
    assert close(S.squared_distance(a, b), S.squared_distance(b, a)), "distance must be symmetric"


def random_vectors(n, dim, ties=False):
    if ties:
        return [[float(rng.randint(-2, 2)) for _ in range(dim)] for _ in range(n)]
    return [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(n)]


def t_topk_sort():
    vecs = random_vectors(200, 4)
    ids = [f"V{i:03d}" for i in range(200)]
    q = [0.1, -0.2, 0.3, 0.0]
    got = S.topk_sort(q, vecs, ids, 5)
    brute = sorted(((S.squared_distance(q, v), i) for i, v in enumerate(vecs)))[:5]
    assert [g[0] for g in got] == [ids[i] for _, i in brute], "wrong neighbours or wrong order"
    assert all(close(g[1], d) for g, (d, _) in zip(got, brute)), "returned distances are wrong"


def t_heap_equals_sort():
    for trial in range(60):
        ties = trial % 2 == 0                               # half the trials have many exact ties
        dim = 2 if trial % 3 == 0 else 4
        vecs = random_vectors(150, dim, ties)
        ids = [f"V{i:03d}" for i in range(150)]
        q = random_vectors(1, dim, ties)[0]
        for k in (1, 5):
            a, b = S.topk_sort(q, vecs, ids, k), S.topk_heap(q, vecs, ids, k)
            assert [x[0] for x in a] == [x[0] for x in b], \
                f"heap and sort disagree (k={k}, ties={ties}): {a} vs {b}"


def t_heap_small_input():
    vecs, ids = [[0, 0], [1, 1], [2, 2]], ["A", "B", "C"]
    got = S.topk_heap([0, 0], vecs, ids, 5)
    assert [g[0] for g in got] == ["A", "B", "C"], "with fewer than k readings, return them all in order"


# ---------------------------------------------------------------- Part 3
def t_project():
    y = S.project(pca_mean, pca_mean, proj)
    assert len(y) == 2, "projection must return 2 values"
    assert all(close(v, 0) for v in y), "the PCA mean should project to (0, 0)"
    for j in range(4):
        z = list(pca_mean)
        z[j] += 1.0
        y = S.project(z, pca_mean, proj)
        assert close(y[0], proj[j][0]) and close(y[1], proj[j][1]), \
            f"unit step in feature {j} should project to row {j} of the matrix"


def t_preservation():
    assert S.neighbor_preservation(["a", "b", "c", "d", "e"], ["e", "x", "a", "y", "z"]) == 2
    assert S.neighbor_preservation(["a", "b", "c", "d", "e"], ["a", "b", "c", "d", "e"]) == 5
    assert S.neighbor_preservation(["a", "b", "c", "d", "e"], ["v", "w", "x", "y", "z"]) == 0


# ---------------------------------------------------------------- Part 4
def t_forward_zero_input():
    a, h, y_hat = S.forward([0, 0, 0, 0], nn)
    assert all(close(x, b) for x, b in zip(a, nn["b1"])), "with z = 0 the pre-activations equal b1"
    exp_h = [max(b, 0.0) for b in nn["b1"]]
    exp_y = 1 / (1 + math.exp(-(sum(v * hh for v, hh in zip(nn["v"], exp_h)) + nn["c"])))
    assert close(y_hat, exp_y), f"y_hat for z = 0 should be {exp_y:.6f}, got {y_hat}"


def t_forward_relu():
    for _ in range(200):
        z = [rng.gauss(0, 2) for _ in range(4)]
        a, h, y_hat = S.forward(z, nn)
        assert len(a) == 3 and len(h) == 3, "network has 3 hidden units"
        for ai, hi in zip(a, h):
            assert close(hi, max(ai, 0.0)), "h must equal ReLU(a)"
        assert 0 < y_hat < 1, "sigmoid output must lie strictly between 0 and 1"


def t_gradient_finite_difference():
    eps = 1e-6
    for trial in range(30):
        z = [rng.gauss(0, 1) for _ in range(4)]
        y = trial % 2
        for j in range(3):
            before = copy.deepcopy(nn)
            grad, new = S.output_weight_update(z, y, nn, j)
            assert nn == before, "output_weight_update must not modify the network"
            up, dn = copy.deepcopy(nn), copy.deepcopy(nn)
            up["v"][j] += eps
            dn["v"][j] -= eps
            numeric = (S.bce(S.forward(z, up)[2], y) - S.bce(S.forward(z, dn)[2], y)) / (2 * eps)
            assert abs(grad - numeric) < 1e-5, f"dL/dv_{j+1}: analytic {grad:.6f} vs numeric {numeric:.6f}"
            assert close(new, nn["v"][j] - nn["eta"] * grad), "update must be v_j - eta * gradient"


for name, fn in [
    ("Part 1  local_summary", t_local_summary),
    ("Part 1  merge vs mean of local means (fixture)", t_merge_fixture),
    ("Part 1  merged means on training data", t_merge_real),
    ("Part 2  standardize", t_standardize),
    ("Part 2  squared_distance", t_sqdist),
    ("Part 2  topk_sort", t_topk_sort),
    ("Part 2  topk_heap matches topk_sort (incl. ties)", t_heap_equals_sort),
    ("Part 2  topk_heap with fewer than k readings", t_heap_small_input),
    ("Part 3  project", t_project),
    ("Part 3  neighbor_preservation", t_preservation),
    ("Part 4  forward with z = 0", t_forward_zero_input),
    ("Part 4  forward ReLU / sigmoid ranges", t_forward_relu),
    ("Part 4  gradient vs finite difference", t_gradient_finite_difference),
]:
    check(name, fn)

width = max(len(n) for n, _, _ in results)
for name, status, msg in results:
    print(f"{status:5}  {name.ljust(width)}  {msg}")
passed = sum(s == "PASS" for _, s, _ in results)
print(f"\n{passed}/{len(results)} checks passed.")
if not os.path.exists(os.path.join(ROOT, "python", "results.md")):
    print("Reminder: run python3 python/starter.py to produce python/results.md for submission.")
sys.exit(0 if passed == len(results) else 1)
