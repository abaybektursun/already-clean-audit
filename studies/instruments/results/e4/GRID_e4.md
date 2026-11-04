EXPLORATORY E4 grid (amendment 40, Ward partition chosen after E3's unblinding): 277 cases on 234 neurons, 277 live under zero; exit 0 (full read); bootstrap over neurons 2,000 draws, seed 0; verdict_e4.json at 05414acf1.

| row | prediction | outcome | reading |
|---|---|---|---|
| E4 known number (E3 bank) | median overlap >= 0.9 | 1.0000 | pass |
| E4 R1 r(I_zero, s_proxy) | r(I, s_proxy) 0.5 [0.25, 0.7] -> holds | 0.8079 [0.7588, 0.8487] | EXPLORATORY: the forecast holds |
| E4 R2 fit I_zero on s_true | the slope's interval contains 1.45 | intercept 0.0398 [0.0159, 0.0623], slope 1.5532 [1.4527, 1.6713] | EXPLORATORY: the program's fit does not generalize |
| R2 beside: fit I_mean on s_true (mean, not the written value) | the slope's interval contains 1.45 | intercept 0.0614 [0.0322, 0.0913], slope 1.5095 [1.3944, 1.6354] | EXPLORATORY: the program's fit generalizes |
| r(I_zero, s_true) | r(I, s_true) 0.7 | 0.8682 [0.8413, 0.8947] | reported |
| median I (zero; mean) | 0.28 [0.20, 0.36] | 0.2908 [0.2591, 0.3417]; 0.3267 [0.2760, 0.3631] | reported (SPEC10 0.24 to 0.33) |
| median s_true | 0.12 (the eighth-shared median) | 0.1600 [0.1400, 0.2000] | reported |
| proxy overlap with the true set | 0.64 (s4_sameweights, known) | 0.6900 [0.6699, 0.7000] | reported |
| abar proxy (s_proxy_abar): overlap with the true set; its two sides with each other | none | 0.0700 [0.0600, 0.0700]; 0.8800 [0.8600, 0.8800] | reported |
| r(I_mean, s_proxy); r(I_mean, s_true) | none | 0.7820 [0.7254, 0.8295]; 0.8428 [0.8052, 0.8764] | reported |
