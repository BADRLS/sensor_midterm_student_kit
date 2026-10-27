# Sensor analytics midterm kit

Work in Python **or** C. Complete only the functions marked TODO.

| Folder | Contents |
| --- | --- |
| data/ | train.csv (1,600 readings), test.csv (400), queries.csv (Q1-Q5) |
| params/ | feature_stats.csv, pca_mean.csv, pca_projection.csv (4 x 2), nn_weights.txt |
| python/ | starter.py, check.py (standard library only, Python 3.8+) |
| c/ | sensor.c (parsing, memory, heap helpers and self-tests supplied), Makefile |
| answers.md | Template for your short answers |

Python, from this folder:

    python3 python/starter.py     # writes python/results.md
    python3 python/check.py       # supplied checks

C, from the c/ folder:

    make run                      # writes c/results.md
    make check                    # supplied self-tests
    make debug                    # AddressSanitizer + UBSan build, then runs both

IDs and fault labels must never enter distance calculations or PCA.
