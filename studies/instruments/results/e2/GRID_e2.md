# E2 grid

Bank `<local>/interp_research/studies/instruments/results/e2/bank`; verdict `studies/instruments/results/e2/verdict_e2.json`; exit 4 (gates failed ['room'] or a rule cancelled).

| rule | prediction | outcome | reading |
|---|---|---|---|
| E2 R1 (positive control) | R_E 0.30 → read | 0.317 [0.247, 0.383] | cancelled |
| E2 R2 (recipe) | Q 0.85 → not separated at 0.8 | "gates failed: ['room']" | cancelled |
| E2 R3 (product) | D 0.10 → holds | "gates failed: ['room']" | cancelled |
| E2 R4 (direction) | D 0.15 → refuted | "gates failed: ['room']" | cancelled |
| E2 R5 (dose) | monotone | "gates failed: ['room']" | cancelled |
| E2 R6 (match) | 0.05 → not separated | "gates failed: ['room']" | cancelled |

| arm | rep [95%] | R [95%] | Δloss | KL | accuracy | onset median (first; second) | first-onset median not reached, share of draws |
|---|---|---|---|---|---|---|---|
| base | 0.760 [0.710, 0.803] | n/a [n/a, n/a] | n/a | n/a | 0.047 | 4752.0; 4771.0 | 0.000 |
| E | 0.443 [0.387, 0.500] | 0.317 [0.247, 0.383] | 0.0084 | 0.0150 | 0.110 | None; None | 0.974 |
| D | 0.453 [0.397, 0.510] | 0.307 [0.240, 0.370] | 0.0088 | 0.0133 | 0.113 | None; None | 0.940 |
| A | 0.523 [0.467, 0.577] | 0.237 [0.173, 0.300] | 0.0010 | 0.0078 | 0.107 | 7861.0; 7885.0 | 0.190 |
| P16 | 0.670 [0.617, 0.723] | 0.090 [0.027, 0.150] | 0.0085 | 0.0088 | 0.093 | 5908.0; 5930.0 | 0.000 |
| P64 | 0.507 [0.453, 0.563] | 0.253 [0.190, 0.320] | 0.0082 | 0.0149 | 0.123 | 7868.0; 7890.0 | 0.391 |
| P256 | 0.383 [0.327, 0.433] | 0.377 [0.307, 0.450] | 0.0117 | 0.0224 | 0.147 | None; None | 1.000 |
| R_norm | 0.667 [0.613, 0.720] | 0.093 [0.033, 0.150] | 0.0025 | 0.0034 | 0.093 | 4959.0; 4981.0 | 0.000 |
| R_KL | 0.680 [0.627, 0.730] | 0.080 [0.017, 0.147] | 0.0146 | 0.0156 | 0.100 | 5140.0; 5182.0 | 0.000 |
