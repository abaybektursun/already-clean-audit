# N2V raw examples at zero GPU (freeze half; N12's ten example births)

Banked bf16 values from N12 score-02 (`n12_freeze.jsonl` -> abl; `arrays/tNN/<unit>_pull.npz` -> H). The fp32 columns are PENDING until the job runs. pred = -h G summed over the unit's positions; meas = N12's bf16 window loss change; fd pred = 0.1 pred.

| birth | t_pre | windows | loss terms | pole (rank) | sum abs h | pred bf16 | meas bf16 | meas / pred | fd pred |
|---|---|---|---|---|---|---|---|---|---|
| L03N7574+:c6 | 11 | 5 | 4126 | L03N7574+ (0) | 1.92 | 0.05655 | 0.1248 | 2.21 | 0.00565 |
| L03N7574+:c6 | 11 | 5 | 4126 | L03N3745+ (1) | 0.624 | -0.002272 | -1.486 | 654 | -0.000227 |
| L03N7574+:c6 | 11 | 5 | 4126 | L03N8204- (2) | 0.606 | -0.0223 | -0.3518 | 15.8 | -0.00223 |
| L03N7574+:c6 | 11 | 5 | 4126 | L03N1651- (3) | 0.695 | 0.05032 | 0.2747 | 5.46 | 0.00503 |
| L03N7746+:c0 | 14 | 2 | 3473 | L03N7746+ (0) | 0.591 | 0.004023 | 0.2878 | 71.5 | 0.000402 |
| L03N7746+:c0 | 14 | 2 | 3473 | L03N5771- (1) | 1.71 | -0.1103 | 0.8274 | -7.5 | -0.011 |
| L03N7746+:c0 | 14 | 2 | 3473 | L03N8235+ (2) | 5.52 | -0.02711 | -0.6201 | 22.9 | -0.00271 |
| L03N7746+:c0 | 14 | 2 | 3473 | L03N1326- (3) | 6.91 | -0.1418 | -0.5012 | 3.54 | -0.0142 |
| L06N9844+:c1 | 14 | 5 | 7495 | L06N9844+ (0) | 1.11 | 0.0298 | -0.3481 | -11.7 | 0.00298 |
| L06N9844+:c1 | 14 | 5 | 7495 | L06N6402+ (1) | 1.15 | -0.01653 | -2.038 | 123 | -0.00165 |
| L06N9844+:c1 | 14 | 5 | 7495 | L06N5288- (2) | 1.39 | 0.03695 | -0.7585 | -20.5 | 0.0037 |
| L06N9844+:c1 | 14 | 5 | 7495 | L06N363- (3) | 1.12 | -0.00821 | -0.321 | 39.1 | -0.000821 |
| L16N1206+:c1 | 13 | 9 | 6320 | L16N1206+ (0) | 8.1 | 0.05481 | -0.03357 | -0.613 | 0.00548 |
| L16N1206+:c1 | 13 | 9 | 6320 | L16N2324- (1) | 1.84 | 0.00916 | -1.465 | -160 | 0.000916 |
| L16N1206+:c1 | 13 | 9 | 6320 | L16N7534- (2) | 2.01 | 0.008199 | -0.9324 | -114 | 0.00082 |
| L16N1206+:c1 | 13 | 9 | 6320 | L16N1542+ (3) | 3.38 | 0.01575 | -0.07324 | -4.65 | 0.00157 |
| L20N5782+:c1 | 11 | 15 | 16182 | L20N5782+ (0) | 65.1 | -0.1872 | -1.141 | 6.09 | -0.0187 |
| L20N5782+:c1 | 11 | 15 | 16182 | L20N1814+ (1) | 268 | -0.2194 | 0.2472 | -1.13 | -0.0219 |
| L20N5782+:c1 | 11 | 15 | 16182 | L20N7103+ (2) | 89.8 | -0.193 | -1.573 | 8.15 | -0.0193 |
| L20N5782+:c1 | 11 | 15 | 16182 | L20N4687- (3) | 56.8 | -0.02453 | -2.755 | 112 | -0.00245 |
| L07N8340+:c4 | 13 | 5 | 5905 | L07N8340+ (0) | 0.872 | 0.03054 | -2.666 | -87.3 | 0.00305 |
| L07N8340+:c4 | 13 | 5 | 5905 | L07N6238+ (1) | 2.23 | -0.03788 | -2.055 | 54.2 | -0.00379 |
| L07N8340+:c4 | 13 | 5 | 5905 | L07N5607- (2) | 0.981 | -0.0008643 | -2.687 | 3.11e+03 | -8.64e-05 |
| L07N8340+:c4 | 13 | 5 | 5905 | L07N952+ (3) | 1.03 | -0.005293 | -2.079 | 393 | -0.000529 |
| L10N9052+:c2 | 13 | 5 | 5709 | L10N9052+ (0) | 0.263 | -0.0001362 | -1.128 | 8.28e+03 | -1.36e-05 |
| L10N9052+:c2 | 13 | 5 | 5709 | L10N3510+ (1) | 4.86 | 0.002408 | -1.742 | -724 | 0.000241 |
| L10N9052+:c2 | 13 | 5 | 5709 | L10N3950- (2) | 7.24 | -0.008364 | -2.215 | 265 | -0.000836 |
| L10N9052+:c2 | 13 | 5 | 5709 | L10N9744+ (3) | 7.84 | 0.01005 | -1.991 | -198 | 0.00101 |
| L10N9794+:c2 | 9 | 7 | 8212 | L10N9794+ (0) | 58.7 | -0.06492 | -0.769 | 11.8 | -0.00649 |
| L10N9794+:c2 | 9 | 7 | 8212 | L10N5132- (1) | 37.3 | -0.02396 | -0.09448 | 3.94 | -0.0024 |
| L10N9794+:c2 | 9 | 7 | 8212 | L10N4370+ (2) | 32.6 | -0.02959 | -0.9502 | 32.1 | -0.00296 |
| L10N9794+:c2 | 9 | 7 | 8212 | L10N7197+ (3) | 23.9 | 0.003843 | 2.891 | 752 | 0.000384 |
| L19N7374-:c1 | 12 | 10 | 9078 | L19N7374- (0) | 26.1 | 0.05802 | 1.295 | 22.3 | 0.0058 |
| L19N7374-:c1 | 12 | 10 | 9078 | L19N7298+ (1) | 82.2 | -0.007338 | 0.9913 | -135 | -0.000734 |
| L19N7374-:c1 | 12 | 10 | 9078 | L19N9178+ (2) | 44 | -0.0677 | 0.5619 | -8.3 | -0.00677 |
| L19N7374-:c1 | 12 | 10 | 9078 | L19N2397+ (3) | 48.1 | 0.01337 | 1.778 | 133 | 0.00134 |
| L24N5066-:c1 | 11 | 1 | 784 | L24N5066- (0) | 99.2 | 0.04816 | 0.2004 | 4.16 | 0.00482 |
| L24N5066-:c1 | 11 | 1 | 784 | L24N3879- (1) | 279 | -0.02147 | 0.2993 | -13.9 | -0.00215 |
| L24N5066-:c1 | 11 | 1 | 784 | L24N3071+ (2) | 393 | -0.05982 | 0.2317 | -3.87 | -0.00598 |
| L24N5066-:c1 | 11 | 1 | 784 | L24N234+ (3) | 404 | 0.2252 | 0.3875 | 1.72 | 0.0225 |

Over these 40 pairs: median |pred| 0.02582 nats, median |meas| 0.8799 nats, sign agreement 0.62, median |0.1 pred| 0.00258 nats.

Texts: the first exemplar and the first of the matched pole's top-3 contexts that holds none of the words this program does not print (the index is given).
- L03N7574+:c6: exemplar 0: ` the Absorptivity of Radiant Barriers
477 Noboa, H. L. 1993-08-01 THESIS
St [[ochastic]]  generation of biologically accurate brain networks
478 Aluri, Aravind `; matched pole L03N2224-, context 0: `�を聴いていたのだけれども自宅でしか聴かないのにそれだと重くね？っちゅうことでRaspberry Pi 3を用いた新たなソリューションを� [[�]] �討したのだがカンタンにはいかな`
- L03N7746+:c0: exemplar 1: `d as a separate entity. And this would result in an LLC being taxed as a C [[ corp]]  and subject to double taxation.

S Corps, on the other hand, are generally`; matched pole L03N463-, context 0: `Sometimes, lack of [[ motivation]]  becomes the reason behind living in the comfort zone. For obvious reasons, people do`
- L06N9844+:c1: exemplar 0: `←2019-11-16 2019-11-17 2019-11-18→ ↑2019 ↑all
00:17:14 -!- z [[zo]] 38 has joined.
00:17:30 <zzo38> Now I internet`; matched pole L06N7825+, context 0: ` dog's dinner.
  • In the final issue of Atomic [[ Rob]] o And the Dogs of War, the second half takes a time skip to the`
- L16N1206+:c1: exemplar 0: `ty
 *	Class to take a numbered file after it has been stemmed and sort
 *	it into pseudo groups where [[ identical]]  stems are adjacent
 *	Rob Hooper
 *	02/11/`; matched pole L16N4452+, context 1: ` good lead?” she asked.

    He glanced at her with a mild f [[rown]] . “No...it’s just as inane as the rest of them.”

`
- L20N5782+:c1: exemplar 0: `died shielding her from bullets, plan to address the court.

"He has to pay the [[ consequences]]  for what he did, and justice will be served," Hileman said.

`; matched pole L20N1330-, context 0: `. Choose whichever method comes [[ naturally]]  to you. 

I love creating things digitally while at the same time having`
- L07N8340+:c4: exemplar 0: `, Stevenson KB, Bradley SF, Society for Healthcare [[ Epidemi]] ology Long-Term Care Special Interest Group. Surveillance definitions of infections in long-term`; matched pole L07N965-, context 0: ` [[NASA]] ’s James Webb Telescope captures a stunning image of a ‘lonely’ galaxy

`
- L10N9052+:c2: exemplar 0: `LEWIS: Thank you so much. Thanks for the great chat. Transcript provided by NPR, [[ Copyright]]  NPR.

Ayesha Rascoe is a White House correspondent for NPR. She`; matched pole L10N4199+, context 0: ` chat with random strangers by way of textual [[ content]] , audio, and video. It does not function in a one-on-one chat`
- L10N9794+:c2: exemplar 0: ` await autorRepository.getAutorByAutorId(autorId)
}

export default { createAutor, updateAutor, deleteAutor, [[ get]] Autores, getAutorByAutorId }
<|endoftext|>`; matched pole L10N10180+, context 0: `w!

AllWebMenus WordPress Menu Addin 1.0.8

AllWeb [[Menus]]  WordPress Menu Addin AllWebMenus WordPress Menu Addin 1.0.`
- L19N7374-:c1: exemplar 1: `You are [[ here]] 

Doepfer Dark Time

Step Sequencer
Published August 2013`; matched pole L19N3415+, context 0: ` a Bulldog, won't let GO!

last post
2 years ago
can view
 [[can]]  comment
atom/rss
official fubar blogs
 2 years ago
f`
- L24N5066-:c1: exemplar 0: `l Products Group, Caledonian Cables Ltd, CMP Products, Copper Crouse-Hinds (Eaton [[),]]  Cortem Group, Elsewedy Electric, Emerson Industrial Automation, BARTEC`; matched pole L24N228-, context 0: `on
involves. It involves the whole man, [[ body]] , mind, spirit. I do not think
you can frame an intelligent definition`
