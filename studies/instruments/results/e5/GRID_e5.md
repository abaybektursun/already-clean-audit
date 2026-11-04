E5 grid; exit 0 (full read); bootstrap 2,000 draws over pairs, seed 0; verdict_e5.json at ddd68c3bb.

| row | prediction | outcome | reading |
|---|---|---|---|
| T1 C_T1 (cross_u fall, input hold) | C_T1 0.02 [0.00, 0.05] (cause 1 holds or not separated) | 0.0433 [0.0209, 0.0656] | not separated |
| T1 own_u fall, input hold | 0.40 [0.30, 0.50] | 0.4018 [0.3527, 0.4582] | gate room_T1 (>= 0.3) |
| T1 target layer only: cross_u, own_u | reported | 0.0464 [0.0215, 0.0621]; 0.3903 [0.3405, 0.4560] | reported |
| T2 C_T2 (cross_u fall, marked token only) | C_T2 0.002 [-0.01, 0.02] (the channel is the other positions) | 0.0052 [0.0017, 0.0098] | the channel is the other positions (holds) |
| T2 own_u fall, marked token only | 0.25 [0.15, 0.40] | 0.3793 [0.3432, 0.4289] | gate room_T2 (>= 0.10) |
| free cross_u, own_u (E1's known 0.0741, 0.4737) | known | 0.0741 [0.0534, 0.1019]; 0.4737 [0.4156, 0.5219] | gate known_number |
| T4 N_T4 (E1b's rule on Olmo-3) | N_T4 0.12 [0.06, 0.20] | 0.1276 [0.0858, 0.2204] | the construction is not the difference |
| T5 member write, median / p90 | gemma's median member write above Olmo's | o7 0.8964 / 2.3361 (n 34); g4 0.0873 / 0.7897 (n 189); q4b 0.1129 / 0.6057 (n 61) | reported, no rule: cause 5 not argued against (gemma's median member write is below Olmo's) |
