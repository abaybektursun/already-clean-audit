# N6 raw examples with L1's numbers

Written by `studies/next/n6/samples/raw_examples_l1.py` from `raw_examples_zero_gpu.json` and L1's records (`studies/next/n6/banks/L1-02/samples/raw_examples.jsonl`, `studies/next/n6/banks/L1-02/edits.jsonl`). Age is the main cluster's birth index T_k. c is the Frobenius norm of the rank-one edit (c = min of the two column norms of the pair), kappa = c / ||w||. dL is the mean change of next-token loss over the 196,608 eval targets, fp32, nats per token. KL_eff is the mean KL(base || edited) at the pole's 50 eval exemplars, nats. The random-direction twin is L2's.

## P002 early: L03N6372- (partner L03N7574-)

layer 3, freeze half; age T_5 (partner T_12); adjusted age -3.39; max|a| 0.254; density 2,603,717

c 0.988553 (||w|| 0.994016, kappa 0.9945); **dL -4.4325e-06 nats per token**; E+ 2.67681; KL_eff 3.1959e-05; R 5.076

Strongest T_15 contexts (bank):

- rank 0, a -0.254: …\ufeffDELIMITER $$\n\nDROP PROCEDURE IF EXISTS `ssp`.`ssp_insertar_descripcion` $$\nCREATE PROC **[EDURE]**  `ssp`.`ssp_insertar_descripcion` (nombr…
- rank 1, a -0.2523: …DROP PROCEDURE IF EXISTS `deleteUser`;\nDELIMITER $$\n\nCREATE PROC **[EDURE]**  deleteUser(IN uname VARCHAR(16))\nLANGUA…
- rank 2, a -0.2519: …DELIMITER $$\n\nDROP PROCEDURE IF EXISTS modifica_estoc_actual$$\nCREATE PROC **[EDURE]**  modifica_estoc_actual (ESTOC_ACT INTEGE…

Five eval tokens whose loss changed most under the edit (window, target position, change, base loss):

- w 36, pos 195, Δℓ -0.4613, base 8.208: …was a night watchman at Kerr Murray Foundry and Chaplain Rhodes was a janitor. Both are buried in Lindenwood Cemetery.\n\n **[Mot]**
- w 3, pos 4, Δℓ -0.0881, base 3.372: …<|endoftext|>ium-matter **[horn]**
- w 13, pos 1847, Δℓ +0.0847, base 5.940: …\n\nSo, I had to switch my ESXi host BACK to a 2u system\n\nI then re-organized my rack and added a Ripe Atlas **[ Probe]**
- w 62, pos 1470, Δℓ -0.0696, base 5.631: …)\n    \u20ac37 Gns Bud\n    11 bud\n\n    NS3 SIMULATOR EXPERT to implement my algorithm more details in the inbox\n\n    \u20ac **[112]**
- w 31, pos 105, Δℓ +0.0669, base 3.543: … obesity.\u00a0Geneva: World Health Organization; 1997.\u00a0[Google Scholar]\n\nObesity and reproduction: a committee opinion (asrm **[.org]**

## P002 late: L03N7574- (partner L03N6372-)

layer 3, freeze half; age T_12 (partner T_5); adjusted age +0.36; max|a| 0.2538; density 2,805,344

c 0.988553 (||w|| 0.988553, kappa 1.0000); **dL -1.4079e-06 nats per token**; E+ 2.82804; KL_eff 2.4538e-05; R 5.103

Strongest T_15 contexts (bank):

- rank 0, a -0.2538: …The past decade was the warmest ever on record, showing that global warming is \"continuing **[ unab]** ated,\" according to a new report from NA…
- rank 1, a -0.2517: …0.2% in a quiet session. The energy sector stood out, rising 1.5%. Occidental Petroleum (O **[XY]** :xnys) gained 4.3%, surging for the seco…
- rank 2, a -0.2277: …n Monoxide, CO, Combustible Gas, O2, Oxygen Product Type: Multi Gas Detector Trade Name: O **[xy]** ge<|endoftext|>…

Five eval tokens whose loss changed most under the edit (window, target position, change, base loss):

- w 36, pos 195, Δℓ -0.3268, base 8.208: …was a night watchman at Kerr Murray Foundry and Chaplain Rhodes was a janitor. Both are buried in Lindenwood Cemetery.\n\n **[Mot]**
- w 20, pos 887, Δℓ -0.1411, base 7.211: …inger F, Teasdale TW, Schulsinger H, Feldman PM, Mednick SA.\n\n(1977). In unstable air, an unsaturated **[ parcel]**
- w 41, pos 66, Δℓ +0.0920, base 9.179: …that best meets your needs.\nUser-Friendly Controls\nOperate with ease in difficult terrain with the John Deere 644 G-Tier **[ Mid]**
- w 86, pos 878, Δℓ +0.0707, base 6.646: … Spring/ Nelsonville\n\nAURORA INFO TECH\nBACH PETER\nBAILEY DENTISTRY\nBILODEAU RUTH\nBITIL **[ JO]**
- w 40, pos 275, Δℓ +0.0671, base 6.313: … Koyo Spherical Roller Bearings 146 460 0 22741 29372R Koyo Koyo Thrust Tapered Roller\n\nrijzend **[ew]**

## P037 early: L08N9190- (partner L08N5063-)

layer 8, freeze half; age T_8 (partner T_13); adjusted age -0.30; max|a| 1.108; density 3,665,101

c 0.862909 (||w|| 0.969705, kappa 0.8899); **dL +4.7638e-06 nats per token**; E+ 37.003; KL_eff 1.1286e-04; R 12.2

Strongest T_15 contexts (bank):

- rank 0, a -1.108: … very first edition of Dungeons and Dragons, added by Gary Gygax himself after inspiration **[ struck]**  in the form of a 70s tv show called Kol…
- rank 1, a -1.017: …For Puck Markham, the idea for Camp Lightbulb wasn't an immediate \"A **[ha]** !\" moment, but the slow force of ideas c…
- rank 2, a -0.9811: …as Rosetta Stone, Egyptian mummies and Parthenon sculptures. This museum is founded in 175 **[3]**  and visited around six million visitors…

Five eval tokens whose loss changed most under the edit (window, target position, change, base loss):

- w 85, pos 1257, Δℓ +0.2547, base 4.747: …ay to bring the Bunny Trail back to the community in a COVID-friendly manner,\" says executive director Candice Penny.\n\n\" **[Crystal]**
- w 1, pos 188, Δℓ +0.1952, base 9.361: …is to make sure a safe and pleasant experience, both for her customers and herself.\n\nIt comes with its own set of risks, **[ Dakota]**
- w 36, pos 271, Δℓ -0.1360, base 6.798: …y known as Westfield, owned almost solely by the Rockhill family. Many of the black families lived in this part of town. **[ Rock]**
- w 1, pos 57, Δℓ +0.1358, base 6.349: …out herself, she chose to venture into this untapped and exciting brand-new way of life.\n\nArmed with a burning desire to **[ make]**
- w 1, pos 70, Δℓ +0.1305, base 4.110: …citing brand-new way of life.\n\nArmed with a burning desire to make a living doing something \u201cfor her,\u201d Dakota set out to **[ make]**

## P037 late: L08N5063- (partner L08N9190-)

layer 8, freeze half; age T_13 (partner T_8); adjusted age +0.00; max|a| 1.071; density 4,023,844

c 0.862909 (||w|| 0.862909, kappa 1.0000); **dL +9.7630e-06 nats per token**; E+ 88.6173; KL_eff 3.9579e-05; R 15.28

Strongest T_15 contexts (bank):

- rank 0, a -1.071: …ike finding the first move, finding the exact wood for you. It's a little bit like the Pin **[oc]** chio story because it should be an inter…
- rank 1, a -1.053: …twitter.\n\nAnd then there were the few jokes, joking about calling ghostbusters and the Win **[chest]** ers and telling Louis and I to just forg…
- rank 2, a -0.9699: … % of 2 to 17-year-olds in a diagnosis has been received by the United States of attention **[ deficit]**  hyperactive disorder (ADHD) sometime.\n\n…

Five eval tokens whose loss changed most under the edit (window, target position, change, base loss):

- w 36, pos 195, Δℓ -0.3786, base 8.208: …was a night watchman at Kerr Murray Foundry and Chaplain Rhodes was a janitor. Both are buried in Lindenwood Cemetery.\n\n **[Mot]**
- w 64, pos 839, Δℓ -0.2274, base 2.542: …m start to finish. Good for learning \"smaller\" projects or tasks one at a time.\n\nTopics taught: HTML5, CSS3, Javascript, **[ Regex]**
- w 17, pos 311, Δℓ +0.1966, base 7.691: …eorge R. R. Martin \u2013 The Sworn Sword Audio Book Free\n\nThe graphic novel adaptation of the second of George R.R. Martin\u2019s **[ Dunk]**
- w 21, pos 411, Δℓ +0.1643, base 1.876: … Atypique this summer \u2014 and another sign of interest in this fast-growing category.\n\nAthletic Brewing was founded in 201 **[7]**
- w 82, pos 1730, Δℓ +0.1484, base 4.925: … a reboot of jellyfish ecology as a central part of the ocean system.\u201d\n\nDr. Doyle, Dr. Houghton and Graeme C. Hays of **[ De]**

## P039 early: L15N10390- (partner L15N6531-)

layer 15, freeze half; age T_6 (partner T_10); adjusted age -1.09; max|a| 3.371; density 2,712,524

c 0.953256 (||w|| 0.953256, kappa 1.0000); **dL -2.7154e-06 nats per token**; E+ 208.425; KL_eff 1.9511e-03; R 17.5

Strongest T_15 contexts (bank):

- rank 0, a -3.371: … Afghanistan by 80 percent because it has been unable to get funding for the project.\n\nThe **[ consortium]**  has proposed new terms which would see …
- rank 1, a -2.996: … Securities Services (BOISS), was the custodian bank of an investment fund involved in the **[ scheme]** .\n\nSource link\n\nContinue Reading\n\nReader…
- rank 2, a -2.883: …er swallow the large losses being generated by its plants and would try to sell them.\n\nThe **[ owner]**  of the plants, Tata Steel, has been squ…

Five eval tokens whose loss changed most under the edit (window, target position, change, base loss):

- w 68, pos 617, Δℓ -0.2523, base 2.896: …an to post an even-par 72 Monday afternoon.\n\nOverall, UHV sits in third place as part of a cluster of teams teams at the **[ top]**
- w 72, pos 1529, Δℓ +0.1962, base 3.413: …ers at the mansion.\n\nAllen said she and the governor were both disappointed that Disney abandoned its plans for the park **[ near]**
- w 50, pos 993, Δℓ +0.1916, base 2.115: …he move is official recognition by The UK Dark Sky Discovery partnership of the quality of the night sky over the garden **[ in]**
- w 13, pos 1290, Δℓ +0.1883, base 6.538: …lly got an ever better switch with PoE which means I didn't need a ton of PoE injectors thrown behind the rack. I got an **[ SG]**
- w 7, pos 849, Δℓ -0.1876, base 4.351: …or of Nigeria\u2019s Civil Resource Development and Documentation Centre, said in Uganda, where the 31 May-11 June conference **[ is]**

## P039 late: L15N6531- (partner L15N10390-)

layer 15, freeze half; age T_10 (partner T_6); adjusted age -0.70; max|a| 3.56; density 2,697,547

c 0.953256 (||w|| 0.997021, kappa 0.9561); **dL +1.7588e-06 nats per token**; E+ 187.633; KL_eff 1.2318e-03; R 16.51

Strongest T_15 contexts (bank):

- rank 0, a -3.56: …it, you have to have a business license in the town where you\u2019re going to install the sign **[.]**  So you have to drive to the licensing o…
- rank 1, a -3.079: … not lying because I didn't believe they were lies. But when I state my belief I am lying. **[ But]**  I'm not lying because I believe it to b…
- rank 2, a -2.53: …out yogurt dispenser, but in order to get to the dispenser he has to use the Pencil Eraser **[,]**  which Paula is holding onto. Well, it's…

Five eval tokens whose loss changed most under the edit (window, target position, change, base loss):

- w 48, pos 1259, Δℓ -0.2341, base 6.326: … office this November, and how do voters in Northern Virginia, you know, young Republicans, older Republicans, you know, **[ translate]**
- w 15, pos 1176, Δℓ +0.1329, base 8.370: …ppening, depending on where the aircaft was and the path, there could be obstacles in the way, and then how do we handle **[ return]**
- w 15, pos 1189, Δℓ +0.1056, base 2.002: …re could be obstacles in the way, and then how do we handle return of control? How does the user know they lost control? **[ How]**
- w 63, pos 1462, Δℓ +0.1016, base 11.869: … said about keeping things simple. So, instead of going into a theory on how to increase your deep link percentage, what **[ threshold]**
- w 23, pos 231, Δℓ -0.1006, base 2.701: …ommerce software, fulfillment software, worry about warehousing, customer service, refunds and so on. But that's not all **[.]**

## P020 early: L22N7990- (partner L22N10231+)

layer 22, freeze half; age T_9 (partner T_12); adjusted age -1.86; max|a| 5.616; density 4,985,676

c 0.984135 (||w|| 0.984135, kappa 1.0000); **dL -6.7423e-07 nats per token**; E+ 1592.3; KL_eff 4.3181e-04; R 23.84

Strongest T_15 contexts (bank):

- rank 0, a -5.616: …nts without disturbing the brake line. It\u2019s similar in concept to the spacers Trek uses on **[ its]**  newer Madone.\n\nEven the stem gets speci…
- rank 1, a -5.423: …points.\n    My only problem is that I'm stuck with a pair of brand new DT 350's (sp, disc, **[ ]** 24/24). It was a bargain, I could not le…
- rank 2, a -5.296: …Choose your location and language.\nInternational News\n\nSamsung Announces Launch of the **[ Samsung]**  Galaxy J1 (2016) 4G\n\nApr 22. …

Five eval tokens whose loss changed most under the edit (window, target position, change, base loss):

- w 30, pos 1865, Δℓ -0.1185, base 2.828: …  9. The King Thing\n  10. A Very Slippery Slope\n  11. The Artifact\n  12. The Phoenix **[ has]**
- w 1, pos 59, Δℓ +0.1040, base 3.856: …self, she chose to venture into this untapped and exciting brand-new way of life.\n\nArmed with a burning desire to make a **[ living]**
- w 14, pos 835, Δℓ +0.0976, base 3.382: …or good for AutoCAD?\n\nThe majority of our recommended laptops come with the Intel quad-core i5, which is more affordable **[ compared]**
- w 61, pos 1454, Δℓ +0.0920, base 3.921: … in some cases. If you follow Hintjens' Law, it was inevitable and predictable that this would change.\n\n\u201cThe cost of any **[ technology]**
- w 67, pos 1775, Δℓ +0.0874, base 1.670: …utably\n  12. Read a matrix as a spiral\n  13. Remove duplicates from a sorted list (state machine)\n  14. **[ Merge]**

## P020 late: L22N10231+ (partner L22N7990-)

layer 22, freeze half; age T_12 (partner T_9); adjusted age +0.39; max|a| 5.536; density 5,457,933

c 0.984135 (||w|| 1.00918, kappa 0.9752); **dL +1.4721e-06 nats per token**; E+ 1710.94; KL_eff 7.8852e-04; R 25.96

Strongest T_15 contexts (bank):

- rank 0, a 5.536: … help predict compatibility and boost the likelihood of finding a compatible companion. By **[ examining]**  factors such as common passions, values…
- rank 1, a 5.266: … There Any Safe Hookup Sites That Work\n\nAre There Any Safe Hookup Sites That Work \u2013 Ashley **[ Madison]**  is known as a fast hookup site for peop…
- rank 2, a 5.263: …\nIn their 1987 article, Hazan and Shaver initially developed self-report measures of three **[ attachment]**  styles (secure, avoidant and ambivalent…

Five eval tokens whose loss changed most under the edit (window, target position, change, base loss):

- w 1, pos 70, Δℓ -0.3708, base 4.110: …citing brand-new way of life.\n\nArmed with a burning desire to make a living doing something \u201cfor her,\u201d Dakota set out to **[ make]**
- w 50, pos 863, Δℓ -0.2191, base 2.707: …more\nAdd Project to Favorites\n\nHelp raise money for this important cause by creating a personalized fundraising page for **[ this]**
- w 59, pos 775, Δℓ +0.1457, base 1.664: …ginate from various other parts of the world, then you can alter your server area.\n\nBluehost Hosting Strategies and also **[ Features]**
- w 56, pos 2013, Δℓ -0.1182, base 6.152: …AFLOODS to 90999 to make a $10 donation.<|endoftext|>s\u00f8k opp hvilket som helst ord, som swag:\nuh **[-no]**
- w 1, pos 71, Δℓ -0.1131, base 1.934: …g brand-new way of life.\n\nArmed with a burning desire to make a living doing something \u201cfor her,\u201d Dakota set out to make **[ a]**

## P069 early: L30N6280+ (partner L30N7717-)

layer 30, freeze half; age T_10 (partner T_13); adjusted age -0.04; max|a| 18.86; density 3,398,232

c 0.972781 (||w|| 0.972781, kappa 1.0000); **dL -4.8689e-06 nats per token**; E+ 16599.4; KL_eff 1.4645e-04; R 30.47

Strongest T_15 contexts (bank):

- rank 0, a 18.86: …r Sr2IrO4 has been discovered. In this system, the strong spin-orbit (SO) coupling as well **[ as]**  the Coulomb interaction is crucial for …
- rank 1, a 18.13: …he resonant inelastic scattering (RIXS) experiments have observed higher energy broad peak **[ at]**  ~0.5-0.8 eV, termed SO exciton[…
- rank 2, a 17.29: …tate. Its 5d5 shell splits into half filled Jeff=1/2 and fully-occupied Jeff=3/2 state due **[ to]**  strong SO coupling. The narrow half fil…

Five eval tokens whose loss changed most under the edit (window, target position, change, base loss):

- w 73, pos 185, Δℓ -0.1438, base 3.847: … Oltrarno, San Marco, and Santa Croce. Outside the city center, check out Via Bolengese, Campo di Marte, and Nov **[oli]**
- w 51, pos 1294, Δℓ +0.0592, base 9.001: …indahl, T, Wood, RD\n\nScience 1999\n11937636 A quantitative model of human DNA base excision repair. I. Mechanistic **[ insights]**
- w 27, pos 1002, Δℓ +0.0556, base 1.427: …do 25 free spins casino ye take me red7 slots for. Yourself d any window curtain in addition,yet chose arranged of schok **[of]**
- w 58, pos 301, Δℓ +0.0456, base 9.194: …sic responses that the user can send, or tweak and send.\n\nIt will be available in English first, followed by Spanish and **[\u00a0]**
- w 35, pos 1734, Δℓ -0.0446, base 1.398: …apture Mont St Quentin or P\u00e9ronne. He had been ordered to simply keep touch with the enemy. However, given Field Marshal **[ Sir]**

## P069 late: L30N7717- (partner L30N6280+)

layer 30, freeze half; age T_13 (partner T_10); adjusted age +1.31; max|a| 17.23; density 4,221,789

c 0.972781 (||w|| 1.0074, kappa 0.9656); **dL +1.1198e-06 nats per token**; E+ 15159.8; KL_eff 2.0253e-04; R 33.29

Strongest T_15 contexts (bank):

- rank 0, a -17.23: …till just barely enough for skiing. Here's to hoping the temperatures stay warm enough for **[ snow]**  and maybe all we will need is a warm bo…
- rank 1, a -16.37: …m that floods the market and has proven side effects published including increased risk of **[ blood]**  clotting and prostate cancer risk and t…
- rank 2, a -16.33: … enjoy lots of outdoor activities like hiking, skiing, mountain biking, rafting, climbing, **[ snow]** boarding, etc. You will find great hospi…

Five eval tokens whose loss changed most under the edit (window, target position, change, base loss):

- w 61, pos 1001, Δℓ +0.0927, base 2.157: …\n\nDo you know?\n\nGartner named SAP S/4HANA Cloud as a Leader in its 2022 Magic Quadrant\u2122 report for Cloud ERP for **[ Service]**
- w 33, pos 1205, Δℓ +0.0864, base 0.593: …fer:\n\n- Packed with antioxidants\n- Promote immune system health\n- May reduce inflammation\n- May enhance gut health\n\nSnow **[ F]**
- w 26, pos 941, Δℓ -0.0622, base 3.935: … add the sifted flour and milk to the beetroot mixture, crush the garlic, salt and season with oil. Mix well.\n  4. Put a **[ non]**
- w 66, pos 521, Δℓ +0.0605, base 2.872: … software delivery pipeline, from code to production, by using Git as the single source of truth for all operational and **[ infrastructure]**
- w 31, pos 1009, Δℓ +0.0578, base 5.990: …ificial into stramash. Cholecystokinin, dime's, as soon as littering - unluxated astrally below indivertible subw **[aking]**
