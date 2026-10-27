# Results

## Part 1 - Partitioned analytics

| Partition | Count | Local mean temperature | Local mean vibration | Local mean pressure | Local mean rpm |
|---|---|---|---|---|---|
| P1 | 100 | 50.9350 | 2.1632 | 4.4778 | 1539.7700 |
| P2 | 300 | 53.2273 | 2.4131 | 4.8151 | 1674.6867 |
| P3 | 500 | 58.3762 | 2.8919 | 5.3016 | 1973.9860 |
| P4 | 700 | 66.1840 | 3.7291 | 6.0866 | 2430.8786 |

| Feature | Merged global mean | Mean of local means | Error |
|---|---|---|---|
| temperature | 60.3616 | 57.1806 | -3.1810 |
| vibration | 3.1228 | 2.7993 | -0.3235 |
| pressure | 5.5023 | 5.1703 | -0.3320 |
| rpm | 2090.6194 | 1904.8303 | -185.7891 |

## Parts 2-3 - Exact similarity search and PCA

| Query | Original-space top-5 (ID: sq. dist) | PCA-space top-5 (ID: sq. dist) | Preservation@5 | Heap = sort? |
|---|---|---|---|---|
| Q1 | R1160: 0.032, R0485: 0.083, R0325: 0.097, R0789: 0.102, R0150: 0.112 | (Part 3 not done) | - | yes |
| Q2 | R1671: 0.053, R1843: 0.060, R1564: 0.087, R1210: 0.117, R1548: 0.161 | (Part 3 not done) | - | yes |
| Q3 | R1618: 0.160, R1393: 0.244, R1907: 0.367, R1489: 0.372, R1185: 0.423 | (Part 3 not done) | - | yes |
| Q4 | R1529: 0.096, R1368: 0.102, R0955: 0.107, R1487: 0.108, R1181: 0.133 | (Part 3 not done) | - | yes |
| Q5 | R1966: 0.529, R1715: 0.687, R1666: 0.708, R1503: 0.777, R1863: 0.883 | (Part 3 not done) | - | yes |

[part4 not finished: TODO: forward]

