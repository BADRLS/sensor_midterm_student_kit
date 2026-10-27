/*
 * Sensor analytics midterm - C version (STUDENT STARTER).
 *
 * Build:      make            (from the c/ directory)
 * Run:        make run        writes c/results.md
 * Check:      make check      runs the supplied self-tests (Part 6)
 * Debug:      make debug      builds with AddressSanitizer + UBSan
 *
 * Complete every function between a "STUDENT" banner and its END marker.
 * Everything else (parsing, memory, heap helpers, output, self-tests) is supplied.
 * IDs and fault labels must never enter distance calculations or PCA.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NF 4                 /* features: temperature, vibration, pressure, rpm */
#define NH 3                 /* hidden units */
#define NPC 2                /* PCA components */
#define K 5
#define MAXK 16
#define WORKED_EXAMPLE_ID "R1377"

static const char *FEATURES[NF] = {"temperature", "vibration", "pressure", "rpm"};
static const int PARTITION_SIZES[4] = {100, 300, 500, 700};

/* ======================================================================= supplied types */
typedef struct { char id[16]; double x[NF]; int fault; } Reading;
typedef struct { Reading *r; int n; } Dataset;
typedef struct { double sums[NF]; int count; } Summary;
typedef struct { double W1[NH][NF], b1[NH], v[NH], c, eta; } Net;
typedef struct { double mean[NF]; double proj[NF][NPC]; } Pca;
/* A candidate neighbour: squared distance and row index into the training set. */
typedef struct { double d; int idx; } Cand;

/* ======================================================================= supplied: TODO tracking */
static int todo_hits = 0;
static const char *todo_last = "";
static void not_implemented(const char *name) { if (!todo_hits) todo_last = name; todo_hits++; }

/* ======================================================================= supplied: loading */
static void die(const char *msg, const char *arg) { fprintf(stderr, "error: %s %s\n", msg, arg); exit(1); }

static Dataset load_readings(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) die("cannot open", path);
    char line[512];
    int cap = 256, n = 0, has_fault;
    Reading *r = malloc(sizeof *r * cap);
    if (!r || !fgets(line, sizeof line, f)) die("bad file", path);
    has_fault = strstr(line, "fault") != NULL;
    while (fgets(line, sizeof line, f)) {
        if (n == cap) { cap *= 2; r = realloc(r, sizeof *r * cap); if (!r) die("out of memory", path); }
        Reading *q = &r[n];
        int got = sscanf(line, "%15[^,],%lf,%lf,%lf,%lf,%d", q->id, &q->x[0], &q->x[1], &q->x[2], &q->x[3], &q->fault);
        if (got < 5) continue;
        if (!has_fault) q->fault = -1;
        n++;
    }
    fclose(f);
    return (Dataset){r, n};
}

static void load_feature_csv(const char *path, int ncols, double out[NF][2]) {
    FILE *f = fopen(path, "r");
    if (!f) die("cannot open", path);
    char line[256], name[64];
    if (!fgets(line, sizeof line, f)) die("empty", path);
    for (int j = 0; j < NF; j++) {
        if (!fgets(line, sizeof line, f)) die("too few rows in", path);
        int got = sscanf(line, "%63[^,],%lf,%lf", name, &out[j][0], &out[j][1]);
        if (got != 1 + ncols) die("bad row in", path);
    }
    fclose(f);
}

static void load_stats(const char *path, double means[NF], double stds[NF]) {
    double t[NF][2];
    load_feature_csv(path, 2, t);
    for (int j = 0; j < NF; j++) { means[j] = t[j][0]; stds[j] = t[j][1]; }
}

static void load_pca(const char *mean_path, const char *proj_path, Pca *p) {
    double t[NF][2];
    load_feature_csv(mean_path, 1, t);
    for (int j = 0; j < NF; j++) p->mean[j] = t[j][0];
    load_feature_csv(proj_path, 2, t);
    for (int j = 0; j < NF; j++) { p->proj[j][0] = t[j][0]; p->proj[j][1] = t[j][1]; }
}

static void read_numbers(FILE *f, double *dst, int count, const char *path) {
    for (int i = 0; i < count; i++)
        if (fscanf(f, "%lf", &dst[i]) != 1) die("bad number in", path);
}

static void load_nn(const char *path, Net *nn) {
    FILE *f = fopen(path, "r");
    if (!f) die("cannot open", path);
    char tok[64];
    double dims[2];
    while (fscanf(f, "%63s", tok) == 1) {
        if (tok[0] == '#') { int ch; while ((ch = fgetc(f)) != '\n' && ch != EOF) {} continue; }
        if (!strcmp(tok, "W1")) { read_numbers(f, dims, 2, path); read_numbers(f, &nn->W1[0][0], NH * NF, path); }
        else if (!strcmp(tok, "b1")) { read_numbers(f, dims, 1, path); read_numbers(f, nn->b1, NH, path); }
        else if (!strcmp(tok, "v")) { read_numbers(f, dims, 1, path); read_numbers(f, nn->v, NH, path); }
        else if (!strcmp(tok, "c")) { read_numbers(f, dims, 1, path); read_numbers(f, &nn->c, 1, path); }
        else if (!strcmp(tok, "eta")) { read_numbers(f, dims, 1, path); read_numbers(f, &nn->eta, 1, path); }
        else die("unknown block in", path);
    }
    fclose(f);
}

/* ======================================================================= supplied: heap helpers
 * A size-k MAX-heap of Cand stored in an array. heap[0] is the WORST candidate kept so far:
 * largest distance, with ties broken by the larger row index.                              */
static int cand_worse(Cand a, Cand b) {          /* 1 if a ranks after b */
    if (a.d != b.d) return a.d > b.d;
    return a.idx > b.idx;
}
static int cand_cmp(const void *pa, const void *pb) {   /* qsort comparator: best first */
    Cand a = *(const Cand *)pa, b = *(const Cand *)pb;
    return cand_worse(a, b) - cand_worse(b, a);
}
static void heap_swap(Cand *h, int i, int j) { Cand t = h[i]; h[i] = h[j]; h[j] = t; }
static void heap_sift_up(Cand *h, int i) {
    while (i > 0) { int p = (i - 1) / 2; if (!cand_worse(h[i], h[p])) break; heap_swap(h, i, p); i = p; }
}
static void heap_sift_down(Cand *h, int n, int i) {
    for (;;) {
        int l = 2 * i + 1, r = l + 1, w = i;
        if (l < n && cand_worse(h[l], h[w])) w = l;
        if (r < n && cand_worse(h[r], h[w])) w = r;
        if (w == i) return;
        heap_swap(h, i, w); i = w;
    }
}
/* Append c to a heap of size *n (caller guarantees capacity). */
static void heap_push(Cand *h, int *n, Cand c) { h[*n] = c; heap_sift_up(h, *n); (*n)++; }
/* Overwrite the root (the current worst) with c and restore heap order. */
static void heap_replace_root(Cand *h, int n, Cand c) { h[0] = c; heap_sift_down(h, n, 0); }
/* Debug aid: prints the heap array and checks the max-heap property. */
static void heap_print(const Cand *h, int n) {
    fprintf(stderr, "heap[%d]:", n);
    for (int i = 0; i < n; i++) fprintf(stderr, " (%.4f,#%d)", h[i].d, h[i].idx);
    for (int i = 1; i < n; i++) if (cand_worse(h[i], h[(i - 1) / 2])) fprintf(stderr, "  <-- property broken at %d", i);
    fprintf(stderr, "\n");
}

static double relu(double a) { return a > 0 ? a : 0.0; }
static double sigmoid(double a) { return 1.0 / (1.0 + exp(-a)); }
static double bce(double y_hat, int y) { return -(y * log(y_hat) + (1 - y) * log(1 - y_hat)); }

/* ======================================================================= STUDENT: Part 1 */
/* Per-feature sums and the count for readings r[0..n-1]. */
static Summary local_summary(const Reading *r, int n) {
    Summary s;
    for (int j = 0; j < NF; j++) s.sums[j] = 0.0;
    s.count = n;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < NF; j++) {
            s.sums[j] += r[i].x[j];
        }
    }
    return s;
}

/* Global means = (sum of local sums) / (sum of local counts). */
static void merge_summaries(const Summary *s, int np, double out[NF]) {
    for (int j = 0; j < NF; j++) out[j] = 0.0;
    int total_count = 0;
    for (int i = 0; i < np; i++) {
        total_count += s[i].count;
        for (int j = 0; j < NF; j++) {
            out[j] += s[i].sums[j];
        }
    }
    if (total_count > 0) {
        for (int j = 0; j < NF; j++) {
            out[j] /= total_count;
        }
    }
}

/* The WRONG method, for comparison: unweighted average of the local means. */
static void mean_of_local_means(const Summary *s, int np, double out[NF]) {
    for (int j = 0; j < NF; j++) out[j] = 0.0;
    if (np <= 0) return;
    for (int i = 0; i < np; i++) {
        for (int j = 0; j < NF; j++) {
            if (s[i].count > 0) {
                out[j] += s[i].sums[j] / s[i].count;
            }
        }
    }
    for (int j = 0; j < NF; j++) {
        out[j] /= np;
    }
}

/* ======================================================================= STUDENT: Part 2 */
/* z_j = (x_j - mean_j) / std_j */
static void standardize(const double x[NF], const double means[NF], const double stds[NF], double z[NF]) {
    for (int j = 0; j < NF; j++) {
        z[j] = (x[j] - means[j]) / stds[j];
    }
}

/* Squared Euclidean distance between two dim-length vectors. No square root. */
static double squared_distance(const double *a, const double *b, int dim) {
    double sum = 0.0;
    for (int j = 0; j < dim; j++) {
        double diff = a[j] - b[j];
        sum += diff * diff;
    }
    return sum;
}

/* vecs is n rows of dim doubles, row-major. Write the best k (k <= n) to out[0..k-1],
 * best first, ties broken by smaller index. Score all n rows into a malloc'd Cand array,
 * qsort it with the supplied cand_cmp, copy the first k, free the array. */
static void topk_sort(const double *q, const double *vecs, int n, int dim, int k, Cand *out) {
    for (int i = 0; i < k; i++) out[i] = (Cand){0.0, -1};
    int count = (k < n) ? k : n;
    if (count <= 0) return;
    Cand *cands = malloc(sizeof(Cand) * n);
    if (!cands) die("out of memory in topk_sort", "");
    for (int i = 0; i < n; i++) {
        cands[i].d = squared_distance(q, vecs + (size_t)i * dim, dim);
        cands[i].idx = i;
    }
    qsort(cands, n, sizeof(Cand), cand_cmp);
    for (int i = 0; i < count; i++) {
        out[i] = cands[i];
    }
    free(cands);
}

/* Same output as topk_sort, using a size-k max-heap (supplied helpers: heap_push,
 * heap_replace_root, cand_worse). After the scan, sort the k survivors with qsort/cand_cmp. */
static void topk_heap(const double *q, const double *vecs, int n, int dim, int k, Cand *out) {
    for (int i = 0; i < k; i++) out[i] = (Cand){0.0, -1};
    int cap = (k < n) ? k : n;
    if (cap <= 0) return;
    int heap_size = 0;
    for (int i = 0; i < n; i++) {
        Cand c = { squared_distance(q, vecs + (size_t)i * dim, dim), i };
        if (heap_size < cap) {
            heap_push(out, &heap_size, c);
        } else if (cand_worse(out[0], c)) {
            heap_replace_root(out, heap_size, c);
        }
    }
    qsort(out, heap_size, sizeof(Cand), cand_cmp);
}

/* ======================================================================= STUDENT: Part 3 */
/* y_c = sum_j (z_j - mean_j) * proj[j][c],  c = 0, 1 */
static void project(const double z[NF], const Pca *p, double y[NPC]) {
    /* TODO: replace this stub with your implementation. */
    (void)z; (void)p; y[0] = y[1] = 0; not_implemented("project");
}

/* Number of indices present in both top-k lists. */
static int neighbor_preservation(const Cand *orig, const Cand *reduced, int k) {
    /* TODO: replace this stub with your implementation. */
    (void)orig; (void)reduced; (void)k; not_implemented("neighbor_preservation"); return -1;
}

/* ======================================================================= STUDENT: Part 4 */
/* a_i = sum_j W1[i][j] z_j + b1[i];  h_i = relu(a_i);  return y_hat = sigmoid(sum_i v[i] h_i + c) */
static double forward(const double z[NF], const Net *nn, double a[NH], double h[NH]) {
    /* TODO: replace this stub with your implementation. */
    (void)z; (void)nn; for (int i = 0; i < NH; i++) a[i] = h[i] = 0; not_implemented("forward"); return 0.5;
}

/* For output weight v_j (0-based): *grad = (y_hat - y) h_j ; *new_v = v_j - eta * grad.
 * Must not modify nn. */
static void output_weight_update(const double z[NF], int y, const Net *nn, int j, double *grad, double *new_v) {
    /* TODO: replace this stub with your implementation. */
    (void)z; (void)y; (void)nn; (void)j; *grad = *new_v = 0; not_implemented("output_weight_update");
}

/* ======================================================================= supplied: main program */
static double knn_fault_accuracy(const double *train, const int *labels, int n,
                                 const double *test, const int *tl, int nt, int dim) {
    Cand best[K];
    int correct = 0;
    for (int t = 0; t < nt; t++) {
        topk_heap(test + (size_t)t * dim, train, n, dim, K, best);
        int votes = 0;
        for (int i = 0; i < K; i++) votes += best[i].idx >= 0 ? labels[best[i].idx] : 0;
        correct += (votes >= 3) == (tl[t] == 1);
    }
    return (double)correct / nt;
}

static void fmt_list(FILE *o, const Cand *c, const Dataset *tr) {
    for (int i = 0; i < K; i++)
        fprintf(o, "%s%s: %.3f", i ? ", " : "", c[i].idx >= 0 ? tr->r[c[i].idx].id : "?", c[i].d);
}

static int run_report(const char *root) {
    char p[1024], p2[1024];
#define PATH(rel) (snprintf(p, sizeof p, "%s/%s", root, rel), p)
    Dataset train = load_readings(PATH("data/train.csv"));
    Dataset test = load_readings(PATH("data/test.csv"));
    Dataset queries = load_readings(PATH("data/queries.csv"));
    double means[NF], stds[NF];
    Pca pca;
    Net nn;
    load_stats(PATH("params/feature_stats.csv"), means, stds);
    snprintf(p2, sizeof p2, "%s/params/pca_projection.csv", root);
    load_pca(PATH("params/pca_mean.csv"), p2, &pca);
    load_nn(PATH("params/nn_weights.txt"), &nn);
    FILE *o = fopen(PATH("c/results.md"), "w");
    if (!o) die("cannot write", p);
    FILE *outs[2] = {stdout, o};
    fprintf(o, "# Results\n\n");

    /* ---- Part 1 */
    todo_hits = 0;
    Summary s[4];
    int start = 0;
    for (int i = 0; i < 4; i++) { s[i] = local_summary(train.r + start, PARTITION_SIZES[i]); start += PARTITION_SIZES[i]; }
    double g[NF], w[NF];
    merge_summaries(s, 4, g);
    mean_of_local_means(s, 4, w);
    for (int f = 0; f < 2; f++) {
        FILE *u = outs[f];
        fprintf(u, "## Part 1 - Partitioned analytics\n\n");
        if (todo_hits) { fprintf(u, "[part1 not finished: TODO: %s]\n\n", todo_last); continue; }
        fprintf(u, "| Partition | Count | Local mean temperature | Local mean vibration | Local mean pressure | Local mean rpm |\n|---|---|---|---|---|---|\n");
        for (int i = 0; i < 4; i++) {
            fprintf(u, "| P%d | %d |", i + 1, s[i].count);
            for (int j = 0; j < NF; j++) fprintf(u, " %.4f |", s[i].sums[j] / s[i].count);
            fprintf(u, "\n");
        }
        fprintf(u, "\n| Feature | Merged global mean | Mean of local means | Error |\n|---|---|---|---|\n");
        for (int j = 0; j < NF; j++) fprintf(u, "| %s | %.4f | %.4f | %+.4f |\n", FEATURES[j], g[j], w[j], w[j] - g[j]);
    }

    /* ---- Parts 2-3 */
    todo_hits = 0;
    int n = train.n, nt = test.n;
    double *Z = malloc(sizeof *Z * n * NF), *Y = malloc(sizeof *Y * n * NPC);
    double *Zt = malloc(sizeof *Zt * nt * NF), *Yt = malloc(sizeof *Yt * nt * NPC);
    int *lab = malloc(sizeof *lab * n), *tl = malloc(sizeof *tl * nt);
    if (!Z || !Y || !Zt || !Yt || !lab || !tl || n < K || nt < 1) die("out of memory or empty data in", root);
    for (int i = 0; i < n; i++) { standardize(train.r[i].x, means, stds, Z + i * NF); project(Z + i * NF, &pca, Y + i * NPC); lab[i] = train.r[i].fault; }
    for (int i = 0; i < nt; i++) { standardize(test.r[i].x, means, stds, Zt + i * NF); project(Zt + i * NF, &pca, Yt + i * NPC); tl[i] = test.r[i].fault; }
    Cand os[8][K], rs[8][K];
    int pres[8], same[8], total = 0, nq = queries.n < 8 ? queries.n : 8;
    for (int qi = 0; qi < nq; qi++) {
        double zq[NF], yq[NPC];
        Cand oh[K], rh[K];
        standardize(queries.r[qi].x, means, stds, zq);
        project(zq, &pca, yq);
        topk_sort(zq, Z, n, NF, K, os[qi]); topk_heap(zq, Z, n, NF, K, oh);
        topk_sort(yq, Y, n, NPC, K, rs[qi]); topk_heap(yq, Y, n, NPC, K, rh);
        pres[qi] = neighbor_preservation(os[qi], rs[qi], K);
        same[qi] = 1;
        for (int i = 0; i < K; i++) same[qi] &= os[qi][i].idx == oh[i].idx && rs[qi][i].idx == rh[i].idx;
        total += pres[qi];
    }
    double acc_o = knn_fault_accuracy(Z, lab, n, Zt, tl, nt, NF);
    double acc_p = knn_fault_accuracy(Y, lab, n, Yt, tl, nt, NPC);
    int faults = 0;
    for (int i = 0; i < nt; i++) faults += tl[i];
    for (int f = 0; f < 2; f++) {
        FILE *u = outs[f];
        fprintf(u, "\n## Parts 2-3 - Exact similarity search and PCA\n\n");
        if (todo_hits) { fprintf(u, "[parts23 not finished: TODO: %s]\n", todo_last); continue; }
        fprintf(u, "| Query | Original-space top-5 (ID: sq. dist) | PCA-space top-5 (ID: sq. dist) | Preservation@5 | Heap = sort? |\n|---|---|---|---|---|\n");
        for (int qi = 0; qi < nq; qi++) {
            fprintf(u, "| %s | ", queries.r[qi].id); fmt_list(u, os[qi], &train);
            fprintf(u, " | "); fmt_list(u, rs[qi], &train);
            fprintf(u, " | %d/5 | %s |\n", pres[qi], same[qi] ? "yes" : "NO");
        }
        fprintf(u, "\nMean neighbor preservation@5: %.1f/5 (%.0f%%)\n", total / (double)nq, 100.0 * total / (5 * nq));
        fprintf(u, "\nTest-set 5-NN fault accuracy: original %.3f, PCA %.3f, always-'no fault' baseline %.3f\n",
                acc_o, acc_p, 1 - faults / (double)nt);
    }

    /* ---- Part 4 */
    todo_hits = 0;
    const Reading *ex = NULL;
    for (int i = 0; i < nt; i++) if (!strcmp(test.r[i].id, WORKED_EXAMPLE_ID)) ex = &test.r[i];
    if (!ex) die("worked example not found:", WORKED_EXAMPLE_ID);
    double z[NF], a[NH], h[NH], grad[NH], nv[NH];
    standardize(ex->x, means, stds, z);
    double y_hat = forward(z, &nn, a, h);
    double sacc = 0;
    for (int i = 0; i < NH; i++) sacc += nn.v[i] * h[i];
    for (int j = 0; j < NH; j++) output_weight_update(z, ex->fault, &nn, j, &grad[j], &nv[j]);
    int correct = 0;
    for (int i = 0; i < nt; i++) correct += (forward(Zt + i * NF, &nn, a, h) >= 0.5) == (tl[i] == 1);
    forward(z, &nn, a, h);                       /* restore a, h for printing */
    for (int f = 0; f < 2; f++) {
        FILE *u = outs[f];
        if (todo_hits) { fprintf(u, "\n[part4 not finished: TODO: %s]\n", todo_last); continue; }
        fprintf(u, "\n## Part 4 - Worked update for %s (fault = %d)\n\n", ex->id, ex->fault);
        fprintf(u, "z = [%.4f, %.4f, %.4f, %.4f]\n", z[0], z[1], z[2], z[3]);
        fprintf(u, "a = [%.4f, %.4f, %.4f]\nh = [%.4f, %.4f, %.4f]\n", a[0], a[1], a[2], h[0], h[1], h[2]);
        fprintf(u, "output pre-activation s = %.4f;  y_hat = %.4f;  loss = %.4f\n", sacc + nn.c, y_hat, bce(y_hat, ex->fault));
        for (int j = 0; j < NH; j++)
            fprintf(u, "v_%d: dL/dv = (%.4f - %d) x %.4f = %+.4f;  v_new = %.2f - %.2f x %+.4f = %.4f\n",
                    j + 1, y_hat, ex->fault, h[j], grad[j], nn.v[j], nn.eta, grad[j], nv[j]);
        fprintf(u, "\nTest-set accuracy of the supplied network (threshold 0.5): %.3f\n", (double)correct / nt);
    }
    fclose(o);
    free(Z); free(Y); free(Zt); free(Yt); free(lab); free(tl);
    free(train.r); free(test.r); free(queries.r);
    return 0;
#undef PATH
}

/* ======================================================================= supplied: self-tests (Part 6) */
static int passed = 0, failed = 0, todo_count = 0;
static unsigned long long rng_state = 11;
static double urand(void) { rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL; return (rng_state >> 11) * (1.0 / 9007199254740992.0); }
static double gauss(void) { double u = urand() + 1e-12, v = urand(); return sqrt(-2 * log(u)) * cos(6.283185307179586 * v); }
static int close_to(double a, double b) { return fabs(a - b) <= 1e-6 * fmax(1.0, fmax(fabs(a), fabs(b))); }

static void report(const char *name, int ok, const char *msg) {
    if (todo_hits) { printf("TODO   %-48s %s\n", name, todo_last); todo_count++; }
    else if (ok) { printf("PASS   %s\n", name); passed++; }
    else { printf("FAIL   %-48s %s\n", name, msg); failed++; }
    todo_hits = 0;
}

static int self_tests(const char *root) {
    char p[1024], p2[1024];
    snprintf(p, sizeof p, "%s/data/train.csv", root);
    Dataset train = load_readings(p);
    double means[NF], stds[NF];
    Pca pca;
    Net nn;
    snprintf(p, sizeof p, "%s/params/feature_stats.csv", root); load_stats(p, means, stds);
    snprintf(p, sizeof p, "%s/params/pca_mean.csv", root);
    snprintf(p2, sizeof p2, "%s/params/pca_projection.csv", root); load_pca(p, p2, &pca);
    snprintf(p, sizeof p, "%s/params/nn_weights.txt", root); load_nn(p, &nn);
    int ok;

    /* Part 1 */
    Reading fx[4] = {{"T1", {0, 0, 0, 0}, 0}, {"T2", {10, 10, 10, 10}, 1}, {"T3", {10, 10, 10, 10}, 0}, {"T4", {10, 10, 10, 10}, 0}};
    Summary s1 = local_summary(fx, 2);
    ok = s1.count == 2 && close_to(s1.sums[0], 10) && close_to(s1.sums[3], 10);
    report("Part 1  local_summary", ok, "sums or count wrong");
    Summary ss[2] = {local_summary(fx, 1), local_summary(fx + 1, 3)};
    double g[NF], w[NF];
    merge_summaries(ss, 2, g); mean_of_local_means(ss, 2, w);
    ok = 1; for (int j = 0; j < NF; j++) ok &= close_to(g[j], 7.5) && close_to(w[j], 5.0);
    report("Part 1  merge vs mean of local means (fixture)", ok, "expected merged 7.5 and mean-of-means 5.0");
    Summary parts[4]; int start = 0, tot = 0;
    for (int i = 0; i < 4; i++) { parts[i] = local_summary(train.r + start, PARTITION_SIZES[i]); start += PARTITION_SIZES[i]; tot += parts[i].count; }
    merge_summaries(parts, 4, g);
    ok = tot == 1600;
    for (int j = 0; j < NF; j++) { double d = 0; for (int i = 0; i < train.n; i++) d += train.r[i].x[j]; ok &= close_to(g[j], d / train.n); }
    report("Part 1  merged means on training data", ok, "merged means differ from a single-pass mean");

    /* Part 2 */
    double z[NF], x[NF];
    standardize(means, means, stds, z);
    ok = 1; for (int j = 0; j < NF; j++) ok &= close_to(z[j], 0);
    for (int j = 0; j < NF; j++) x[j] = means[j] + 2 * stds[j];
    standardize(x, means, stds, z);
    for (int j = 0; j < NF; j++) ok &= close_to(z[j], 2);
    report("Part 2  standardize", ok, "mean -> 0 and mean + 2 sd -> 2 expected");
    double o0[2] = {0, 0}, o34[2] = {3, 4};
    ok = close_to(squared_distance(o0, o34, 2), 25) && close_to(squared_distance(o34, o34, 2), 0);
    report("Part 2  squared_distance", ok, "d^2((0,0),(3,4)) should be 25");

    int nv = 150;
    double *vecs = malloc(sizeof *vecs * nv * NF);
    Cand a5[MAXK], b5[MAXK];
    ok = 1;
    for (int trial = 0; trial < 60 && ok; trial++) {
        int dim = trial % 3 == 0 ? 2 : 4, ties = trial % 2 == 0;
        for (int i = 0; i < nv * dim; i++) vecs[i] = ties ? (double)((int)(urand() * 5) - 2) : gauss();
        double q[NF];
        for (int j = 0; j < dim; j++) q[j] = ties ? (double)((int)(urand() * 5) - 2) : gauss();
        for (int k = 1; k <= 5; k += 4) {
            topk_sort(q, vecs, nv, dim, k, a5);
            topk_heap(q, vecs, nv, dim, k, b5);
            for (int i = 0; i < k; i++) {
                double d = squared_distance(q, vecs + a5[i].idx * dim, dim);
                ok &= a5[i].idx >= 0 && a5[i].idx == b5[i].idx && close_to(a5[i].d, d);
                if (i) ok &= !cand_worse(a5[i - 1], a5[i]);
            }
            /* brute-force check of the k-th best */
            int better = 0;
            for (int i = 0; i < nv; i++) { Cand c = {squared_distance(q, vecs + i * dim, dim), i}; better += cand_worse(a5[k - 1], c); }
            ok &= better == k - 1;
        }
    }
    report("Part 2  topk_sort / topk_heap agree (incl. ties)", ok, "heap and sort disagree, or results not the true best k");
    free(vecs);

    /* Part 3 */
    double y[NPC];
    project(pca.mean, &pca, y);
    ok = close_to(y[0], 0) && close_to(y[1], 0);
    for (int j = 0; j < NF; j++) {
        double zz[NF]; memcpy(zz, pca.mean, sizeof zz); zz[j] += 1;
        project(zz, &pca, y);
        ok &= close_to(y[0], pca.proj[j][0]) && close_to(y[1], pca.proj[j][1]);
    }
    report("Part 3  project", ok, "PCA mean -> (0,0); unit step in feature j -> row j");
    Cand A[5] = {{0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}}, B[5] = {{0, 5}, {0, 9}, {0, 1}, {0, 8}, {0, 7}};
    ok = neighbor_preservation(A, B, 5) == 2 && neighbor_preservation(A, A, 5) == 5;
    report("Part 3  neighbor_preservation", ok, "expected 2 and 5 on fixtures");

    /* Part 4 */
    double a[NH], h[NH], zero[NF] = {0};
    double yh = forward(zero, &nn, a, h), sacc = nn.c;
    ok = 1;
    for (int i = 0; i < NH; i++) { ok &= close_to(a[i], nn.b1[i]); sacc += nn.v[i] * relu(nn.b1[i]); }
    ok &= close_to(yh, sigmoid(sacc));
    for (int t = 0; t < 200; t++) {
        for (int j = 0; j < NF; j++) z[j] = 2 * gauss();
        yh = forward(z, &nn, a, h);
        ok &= yh > 0 && yh < 1;
        for (int i = 0; i < NH; i++) ok &= close_to(h[i], relu(a[i]));
    }
    report("Part 4  forward", ok, "check a = W1 z + b1, h = ReLU(a), sigmoid output");
    ok = 1;
    for (int t = 0; t < 30; t++) {
        for (int j = 0; j < NF; j++) z[j] = gauss();
        int yy = t % 2;
        for (int j = 0; j < NH; j++) {
            Net before = nn, up = nn, dn = nn;
            double gr, nw, eps = 1e-6;
            output_weight_update(z, yy, &nn, j, &gr, &nw);
            ok &= memcmp(&before, &nn, sizeof nn) == 0;
            up.v[j] += eps; dn.v[j] -= eps;
            double num = (bce(forward(z, &up, a, h), yy) - bce(forward(z, &dn, a, h), yy)) / (2 * eps);
            ok &= fabs(gr - num) < 1e-5 && close_to(nw, nn.v[j] - nn.eta * gr);
        }
    }
    report("Part 4  gradient vs finite difference", ok, "analytic gradient or update rule wrong");

    printf("\n%d/%d checks passed%s.\n", passed, passed + failed + todo_count, todo_count ? " (some still TODO)" : "");
    free(train.r);
    return failed || todo_count;
}

int main(int argc, char **argv) {
    const char *root = "..";                       /* kit root, relative to c/ */
    int selftest = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--selftest")) selftest = 1;
        else root = argv[i];
    }
    (void)heap_print;                              /* available for debugging */
    (void)not_implemented; (void)cand_cmp; (void)heap_push; (void)heap_replace_root; (void)cand_worse;
    return selftest ? self_tests(root) : run_report(root);
}
