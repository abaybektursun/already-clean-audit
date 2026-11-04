E1b grid; exit 0 (full read); bootstrap 2,000 draws over pairs, seed 0; verdict_e1b.json at 1c31d48a0.

| row | prediction | outcome | reading |
|---|---|---|---|
| E1b R5 google/gemma-3-4b-pt | G(mean) 0.08 [0.03, 0.15]; D1 above 0.04 -> holds (§3.5, §14) | G(mean) 0.0050 [-0.0264, 0.0419]; D1 -0.0052 [-0.0226, 0.0297], line 0.0025 | no term to remove (G(mean) 0.0050 < 0.03) |
| R5z, R5m g4 | reported beside R5 | D1z 0.0932 [0.0010, 0.2584]; D1m -0.0040 [-0.0176, 0.0250] | no term to remove (G(mean) 0.0050 < 0.03); no term to remove (G(mean) 0.0050 < 0.03) |
| A1 g4 N(v, u) / N(v, e) | the false-positive line per value | mean 0.0042 / -0.0008 (p95 0.2355 / 0.0600); zero -0.0897 / -0.0014 (p95 0.7115 / 0.0650); median 0.0092 / 0.0002 (p95 0.2302 / 0.0613); res_mean 0.0136 / 0.0035 (p95 0.2131 / 0.0657) | reported |
| regime split g4 | amendment 6: D1 carried by the sparse members; on dense-member sets the zero arm's null is the highest of the four | sparse n 1 D1 0.0021, zero highest; dense n 25 D1 -0.0113, median highest; mixed n 29 D1 -0.0101, res_mean highest; none n 1 D1 -0.0004, zero highest | reported |
| high-background members g4 | Olmo: 6.1 percent of members, median 3 percent of attribution (§3.6) | 293 distinct, 0.1271 of set members, median 0.1175 of attribution, median layer 17.0000 | reported |
| instrument gates g4 | own >= 0.3, random < 0.5 own, cut >= 10 x median in >= half the layers | own 0.6474, random 0.0069, high-background share 0.7879 | pass |
| E1b R5 Qwen/Qwen3-4B-Base | G(mean) 0.06 [0.02, 0.12]; D1 above 0.03 -> holds (§3.5, §14) | G(mean) 0.0023 [-0.0003, 0.0051]; D1 0.0007 [-0.0026, 0.0042], line 0.0012 | no term to remove (G(mean) 0.0023 < 0.03) |
| R5z, R5m q4b | reported beside R5 | D1z 0.0022 [-0.0007, 0.0048]; D1m 0.0015 [-0.0011, 0.0038] | no term to remove (G(mean) 0.0023 < 0.03); no term to remove (G(mean) 0.0023 < 0.03) |
| A1 q4b N(v, u) / N(v, e) | the false-positive line per value | mean 0.0033 / 0.0009 (p95 0.1583 / 0.1583); zero -0.0014 / -0.0015 (p95 0.1608 / 0.1608); median 0.0010 / 0.0002 (p95 0.1550 / 0.1550); res_mean 0.0066 / 0.0050 (p95 0.1570 / 0.1570) | reported |
| regime split q4b | amendment 6: D1 carried by the sparse members; on dense-member sets the zero arm's null is the highest of the four | sparse n 3 D1 0.0035, res_mean highest; dense n 4 D1 0.0007, res_mean highest; mixed n 9 D1 0.0019, res_mean highest; none n 40 D1 0.0000, res_mean highest | reported |
| high-background members q4b | Olmo: 6.1 percent of members, median 3 percent of attribution (§3.6) | 108 distinct, 0.0213 of set members, median 0.0000 of attribution, median layer 29.0000 | reported |
| instrument gates q4b | own >= 0.3, random < 0.5 own, cut >= 10 x median in >= half the layers | own 0.5700, random -0.0009, high-background share 0.5143 | pass |
