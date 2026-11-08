# M2 ten raw examples, zero-GPU part (freeze half)

Written by `python -m infra.pg.m2 examples` from the laptop banks; the rule is in `infra/pg/m2_examples.py`. Units: a+ in raw MLP activation units; q999 is the carrier's banked 99.9th percentile at that point (a histogram bin edge). The text marks the activating token as [[ ]].

## L03N7574+:c6 — miscellaneous code/text subword tokens

Size 5, born T_12 (step 320,000); scored at T_11 (step 180,000); p_late 0.0040; main False.

- Carrier at T_11: median a+ over the exemplars 0.02517 against q999 0.1496 (ratio 0.168); percentile bracket below 0.99; max|a| 1.354.
- Carrier at T_12: median 0.3125, q999 0.1585, ratio 1.971, present True.
- A_own (exemplar positions in the carrier's own T_11 top-50): 1; rank under A among the layer's 20 grid poles: 1; rank among 22,016: PENDING: needs every pole's top-50 at T_(k-1), S3 results/neurons/w3/grid/1b/out/t11_step180000/topk/layer_03.npz. Scored: PENDING.
- Top-10 lists under A and B: PENDING: job m2_score writes out/m2/samples/raw_examples_freeze.jsonl.
- Pseudo-birth twin: L03N4280- ranks [7, 21, 26, 37, 49], ratio at T_11 0.000, A_own 0

Exemplars (T_15 contexts; a+ at T_(k-1) and T_k):

- rank 6 (a+ 0.503 → 0.774): `'ation using synthetic jets\n476 Kim, Kihwan 2005-12-01 THESIS\nThe Influence of Dust on the Absorptivity of Radiant Barriers\n477 Noboa, H. L. 1993-08-01 THESIS\nSt [[ochastic]]  generation of biologically accurate brain networks\n478 Aluri, Aravind '`
- rank 8 (a+ 0 → 0): `'import numpy as np\nimport glob\nimport os\nimport yaml\n\nfrom [[ shutil]]  import copy, rmtree\n\nfrom ..qtim_preprocessing.motion_correction import motion'`
- rank 17 (a+ 0.0252 → 0.312): `'n the 1950s and 1964 by Corneliu E. Albego, Limpidon and Paxor. Patti Scialfa also provided backing vocals on where to purchase sibutramine online india both. B [[àng]]  National Park, has discovered 1,320 additional species in this park, of'`
- rank 26 (a+ 1.17 → 0.984): `'ar subfield, I deal mostly with atoms and their interaction with electromagnetic fields. A few decades ago, a lot of work was done to understand how to cool bos [[onic]]  atoms to within billionths of absolute zero (this is known as a Bose-E'`
- rank 31 (a+ 0 → 0.0541): `'e “high tech” industry partners combining seamless technical support inside the NEVONEX system, adds Saik.\n\nWake up with RealAgriculture\n\nSubscribe to our daily [[ newsletters]]  to keep you up-to-date with our latest coverage every morning.\n\nWake up with'`

Carrier's own top-5 at T_11:

- rank 0 (a 1.35): `'heet intersects itself, modulo two, is included in the action (this term had been written down earlier by Balachandran et. al. [12]). It appears that even class [[ically]] , certain solutions are unstable [13]. Nonetheless, the presence of such terms may'`
- rank 1 (a 1.27): `' [[Classic]]  New York City Taxi\n\nOld fashioned New York City Taxi.\n\nI saw this old'`
- rank 2 (a 1.25): `'n of the individual stars or the orbital parameters of the binary system. The answer is in both cases yes, but the former is much more relevant than the latter. [[ Stellar]]  rotation is tightly linked to flaring: All else equal, the faster the star'`
- rank 3 (a 1.2): `'ature dependence seems to be present after all (or if it is present, it is numerically small). The basic method used in references [7], [8], [9] is a version of [[ classical]]  perturbation theory.\n\nThe presence of terms quadratic in the extrinsic curvature tensor in'`
- rank 4 (a 1.18): `'l be armed with the knowledge to choose the right engine for you or help your customer to make the right decision for them.\n\nVintage Engines: Rebuilt to Drive\n\n [[Classic]]  and vintage car culture in is a diverse one. It’s more than just muscle'`

Matched random-pole reading (three other grid poles of the layer nearest in log max|a| at T_11), read beside the carrier:

- L03N9631- (max|a| 1.38, A on the exemplars 0):
  - rank 0 (a -1.38): `'f Africa. Africa has a large quantity of natural resources, including diamonds, sugar, salt, gold, iron, cobalt, uranium, copper, bauxite, silver, petroleum and [[ cocoa]]  beans, but also woods and tropical fruits. Having a low human density, for'`
  - rank 1 (a -1.29): `'ed” by a witchdoctor often suffer stigma, discrimination and ridicule. Three other victims we spoke to had similar experiences.\n\nThe Centre for Human Rights and [[ Rehabilitation]] ’s (CHRR) acting executive director, Michael Kaiyatsa, said'`
  - rank 2 (a -1.25): `'nted at all due to the government delaying or not releasing counterpart funds. Secondly, most funds for development projects to meet specific policy targets are [[ donor]]  oriented and beyond the government’s control. Thirdly, there is a lack of'`
- L03N1487- (max|a| 1.446, A on the exemplars 0):
  - rank 0 (a -1.45): `'ght for a authoritative Internet of educating out which one it saves. initiative, The Turn of the careamus - Page creatively to the Henry James humum. The Jolly [[ Corner]] ( html, Gaslight, Mt. huge: Below the Henry James instance sol'`
  - rank 1 (a -1.24): `'inai who issued his declarations to Moses, the Ark of the Covenant was to be constructed and used against Siva who had now taken back Jerusalem. Many of the str [[agg]] ling Hyksos were killed by Moses as they were harassing the Nibiru'`
  - rank 2 (a -1.22): `'here was a large pallet of insulation on the floor and an enormous furnace swaddled in the stuff against the far wall. It was damp and rotten and hanging in str [[agg]] les from the pipes. To the side, there were a number of split and'`
- L03N7746- (max|a| 1.249, A on the exemplars 0):
  - rank 0 (a -1.25): `'mily justice on the Canada Revenue Agency site.\n\n**Parenting After Divorce**\n\nThe page on Parenting After Divorce contains some of the same links found in Child [[ Support]] , and also has a link to Resources for Children, which is a user-friendly'`
  - rank 1 (a -1.18): `' which multinational families are no longer exceptional, are increasing in number exponentially. The new Hague Convention on the International Recovery of Child [[ Support]]  and Other Forms of Family Maintenance is designed to respond to the often modest needs of'`
  - rank 2 (a -1.1): `'ovides a direct link to three websites: Child Support, Family Violence and Parenting After Divorce. Here’s an example of what you’ll find on each site:\n\n**Child [[ Support]] **\n\nOn the Child Support page you will find direct links to the Federal Child Support'`

## L03N7746+:c0 — C corp / C-corporation completion

Size 5, born T_15 (step 1,413,814); scored at T_14 (step 1,000,000); p_late 0.0005; main True.

- Carrier at T_14: median a+ over the exemplars 0.06587 against q999 0.07499 (ratio 0.878); percentile bracket [0.99, 0.999); max|a| 0.4715.
- Carrier at T_15: median 0.1227, q999 0.03758, ratio 3.263, present True.
- A_own (exemplar positions in the carrier's own T_14 top-50): 2; rank under A among the layer's 20 grid poles: 1; rank among 22,016: PENDING: needs every pole's top-50 at T_(k-1), S3 results/neurons/w3/grid/1b/out/t14_step1000000/topk/layer_03.npz. Scored: PENDING.
- Top-10 lists under A and B: PENDING: job m2_score writes out/m2/samples/raw_examples_freeze.jsonl.
- Pseudo-birth twin: none found

Exemplars (T_15 contexts; a+ at T_(k-1) and T_k):

- rank 0 (a+ 0.234 → 0.18): `' the personal income tax level\n\nFor tax purposes, LLCs and S Corps are both treated as pass-through entities – avoiding the headache of double taxation like a C [[ corp]] . Yet, the amount of tax a business pays varies depending on the way it'`
- rank 3 (a+ 0.211 → 0.165): `'s. While LLCs are most times taxed applying personal rates, some owners opt to be taxed as a separate entity. And this would result in an LLC being taxed as a C [[ corp]]  and subject to double taxation.\n\nS Corps, on the other hand, are generally'`
- rank 12 (a+ 0.0659 → 0.123): `' you to form as a C-corporation for them to comply with Section 1202 of the tax code (which is set to expire in 2011, so this matter may become moot).\xa0 If a C-c [[orpor]] ation is required by your investors,\xa0it is easy enough to transition from an'`
- rank 14 (a+ 0.0425 → 0.122): `'LLC directly to the shareholders.\xa0 That said, if you anticipate raising outside capital from venture capitalists, many of them will require you to form as a C-c [[orpor]] ation for them to comply with Section 1202 of the tax code (which'`
- rank 40 (a+ 0.0362 → 0.108): `'LC down the road.\xa0 If a financing is not imminent, the tax\xa0savings you will realize from the LLC is typically greater than the legal costs\xa0of switching to a C-c [[orpor]] ation down the road.\xa0 So, more often than not, it is best'`

Carrier's own top-5 at T_14:

- rank 0 (a 0.471): `'y were trying to convince us to join another software. MEXC has proven to me to be highly reliable and a powerful part of my trading arsenal. Registering on Bit [[stamp]]  is incredibly easy to do. Check out the MEXC Creators program.'`
- rank 1 (a 0.417): `', and Banxa. Open your BingX account, and select Deposit. This crypto trading robot takes advantage of the market volatility to trade on behalf of the user. Bit [[stamp]]  is not a wallet, but it does provide a secure platform for buying and selling'`
- rank 2 (a 0.299): `' to bookmark his whole list of post on WordPress Membership plugin alternatives. Another affiliate management program that’s really simple to start with is Tapf [[iliate]] . Optionally, you can utilize Shareasale. With any of these programs,'`
- rank 3 (a 0.299): `'inish with it? Please do share.\n\nyour story. If I were you, I ‘d simply use Unbounce or Leadpages in combination with OptinMonster, Drip, and PartnerStack( Tapf [[iliate]]  or ShareaSale). Let’s Work Together!Whether you’re searching for aid'`
- rank 4 (a 0.296): `'argument is the bitwise OR of zero or more of the following values:\n\nMS_ASYNC\tPerform asynchronous writes.\nMS_SYNC\t\tPerform synchronous writes.\nMS_INVALIDATE\tIn [[validate]]  cached data after writing.\n\nUpon successful completion, the value\xa00 is returned;'`

Matched random-pole reading (three other grid poles of the layer nearest in log max|a| at T_14), read beside the carrier:

- L03N10451- (max|a| 0.4658, A on the exemplars 0):
  - rank 0 (a -0.466): `'oday? Back in 2007 we reviewed a book by SEIU leader Andy Stern called\xa0Getting America back on track: a country that works. Andy Stern and Teamsters’ boss Jimmy [[ Hoff]] a led a 40% split in the AFL-CIO in 2005'`
  - rank 1 (a -0.462): `'Investors of Every Size Have Embraced Investing Alternatives\n\nbank of america merril l [[ynch]]  at 100 newgate streetThe significant shift by a growing number of investors,'`
  - rank 2 (a -0.442): `'from typing import List\n\n\nclass Solution:\n    def validTicTac [[Toe]] (self, board: List[str]) -> bool:\n        cnt1, cnt2'`
- L03N1487+ (max|a| 0.5028, A on the exemplars 0):
  - rank 0 (a 0.503): `'aterfall and his observer Lt. Charles George Gordon Bayly, of 5 Squadron—flying an Avro 504 over Belgium, were killed by infantry fire. On the same day, Captain [[ Charl]] ton (observer) and his pilot, Lieutenant Wadham, made the crucial'`
  - rank 1 (a 0.446): `'cost, to which the Charity Commissioners agreed, and the Board of Works contributed £25,000. The parish of Stoke Newington gave £10,000 ; South Hornsey, £6,000; [[ Hack]] ney, £5,000; and Islington, £2,500—'`
  - rank 2 (a 0.428): `'etropolitan Board of Worke giving £25,000, the Charity Commissioners £47,500, the Vestry\xa0of Stoke Newington £10,000, South Hornsey £6,000, the District Board of [[ Hack]] ney £5,000\xa0and the Islington Vestry £2,500'`
- L03N7574+ (max|a| 0.5058, A on the exemplars 0):
  - rank 0 (a 0.506): `'tential of sorghum [Sorghum bicolor (L.) Moench] adapted to the southern Africa region\n489 Mpofu, Leo Thokoza 2005-12-01 THESIS\nSensitivity of mesoscale gravity [[ waves]]  to the baroclinicity of jet-front systems\n490 Wang, Shugu'`
  - rank 1 (a 0.505): `'er spectrum. We examine magnetoacoustic wave propagation and mode conversion from the photosphere to the corona. Frequency filtering of the upwardly propagating [[ acoustic]]  waves is a natural consequence of a gravitationally stratified atmosphere, and plays'`
  - rank 2 (a 0.477): `'n the 1950s and 1964 by Corneliu E. Albego, Limpidon and Paxor. Patti Scialfa also provided backing vocals on where to purchase sibutramine online india both. B [[àng]]  National Park, has discovered 1,320 additional species in this park, of'`

## L06N9844+:c1 — IRC logs: zzo38 and backslash tokens

Size 7, born T_15 (step 1,413,814); scored at T_14 (step 1,000,000); p_late 0.0270; main False.

- Carrier at T_14: median a+ over the exemplars 0.08258 against q999 0.1189 (ratio 0.695); percentile bracket [0.99, 0.999); max|a| 0.5425.
- Carrier at T_15: median 0.2044, q999 0.06683, ratio 3.058, present True.
- A_own (exemplar positions in the carrier's own T_14 top-50): 1; rank under A among the layer's 20 grid poles: 1; rank among 22,016: PENDING: needs every pole's top-50 at T_(k-1), S3 results/neurons/w3/grid/1b/out/t14_step1000000/topk/layer_06.npz. Scored: PENDING.
- Top-10 lists under A and B: PENDING: job m2_score writes out/m2/samples/raw_examples_freeze.jsonl.
- Pseudo-birth twin: L06N10970- ranks [25, 28, 37, 38, 40, 46, 49], ratio at T_14 0.990, A_own 0

Exemplars (T_15 contexts; a+ at T_(k-1) and T_k):

- rank 1 (a+ 0.0826 → 0.249): `'←2019-11-16 2019-11-17 2019-11-18→ ↑2019 ↑all\n00:17:14 -!- z [[zo]] 38 has joined.\n00:17:30 <zzo38> Now I internet'`
- rank 3 (a+ 0.434 → 0.23): `'ewline?\n22:00:37 * kingoffrance ducks and covers and rolls\n22:03:05 <b_jonas> kingoffrance: no, but geordi, which used to be a bot that evaluated C++, used back [[slash]]  to represent newlines in code you give in irc\n22:03:06'`
- rank 6 (a+ 0.0403 → 0.214): `' certainly in Befunge.\n19:06:35 <fungot> fizzie: think about the data\n19:06:40 <kspalaiologos> is it?\n19:06:44 <kspalaiologos> well\n19:06:46 <fizzie> z [[zo]] 38: CologneBlue should be back, if you want to check.\n19:'`
- rank 11 (a+ 0.282 → 0.204): `"they only represent a newline when it's outside of a string/character literal\n22:15:50 -!- arseniiv has quit (Ping timeout: 276 seconds).\n22:33:56 <ais523> back [[slash]] -space makes a good way to represent newlines without worrying about parsing\n22:"`
- rank 20 (a+ 0.265 → 0.19): `'ew style digit separators to do enough of the tokenization to recognize character literals correctly\n22:38:53 <b_jonas> oh wait\n22:38:57 <b_jonas> you said back [[slash]] -space\n22:38:57 <b_jonas> hmm\n22:'`
- rank 21 (a+ 0.000213 → 0.19): `'ands push themself instead, and a separate command makes them active.\n07:35:19 <esowiki> [[EsoPost]] https://esolangs.org/w/index.php?diff=67135&oldid=67072 * Z [[zo]] 38 * (+1141) EsoPost II variant, and some more explanations'`
- rank 22 (a+ 0 → 0.19): `"s returned.\n05:14:22 <esowiki> [[Quine]] https://esolangs.org/w/index.php?diff=67126&oldid=67124 * Quadril-Is * (-43) /* How to write quines */\n05:17:03 <kmc> z [[zo]] 38: I don't know\n05:17:06 <kmc>"`

Carrier's own top-5 at T_14:

- rank 0 (a 0.542): `'ting\n13 September – 24 September 2021\nEPSC Abstracts\nVol. 15, EPSC2021-414, 2021\nhttps://doi.org/10.5194/epsc2021-414\nEuropean Planetary Science Congress 2021\n© [[ Author]] (s) 2021. This work is distributed under\nthe Creative Commons Attribution'`
- rank 1 (a 0.541): `'worry if you can’t name them all.\n\nOnce you’ve named most of the plates, draw arrows to show the general plate motions. Finally, using a highlighter or coloured [[ pencil]] , label as many of the boundaries as you can as divergent, convergent'`
- rank 2 (a 0.459): `'aman Belial, Hikari begins developing the Ultra Capsules and Risers in order to turn the tides of the battle. Unfortunately before they can be utilized, Kei Fuk [[uide]]  stole them and converted most of its mechanism into Kaiju Capsules.\n\nIn the'`
- rank 3 (a 0.452): `'and Acrylics Colour mixing blue accurately can seem tricky as there are many types of blues to There is the violet-blue of a summer sky; powder-blue of the corn [[flower]] ; How do I make oil painting cheap?. Full standings color mixing oil paints with'`
- rank 4 (a 0.446): `'detection (by flying monster).\n\nHannah Höch, With Seaweed, 1950\nCut-and-pasted papers, torn papers, and gouache on paper, 13 5/8 x 9 7/8 inches\n\nHannah Höch, Gl [[ued]]  Drawing II, 1955\nPhotomontage, 14 x '`

Matched random-pole reading (three other grid poles of the layer nearest in log max|a| at T_14), read beside the carrier:

- L06N4452+ (max|a| 0.5566, A on the exemplars 0):
  - rank 0 (a 0.557): `'\nBiography of George B. Keeler\n\nSearch [[ Fold]] 3 for your\nNative American Records\n\nThere is no man who has taken more'`
  - rank 1 (a 0.545): `'Writing Football News for a Blog\n\nเ [[ท]] คโนโลยีที่ปลอด�'`
  - rank 2 (a 0.514): `'\xa0Sorry for the [[ inconvenience]] .\xa0 There may be some Navigation header breakage of “Previous:” and'`
- L06N2537- (max|a| 0.5795, A on the exemplars 0):
  - rank 0 (a -0.58): `'rces (or friends of sources) mentioning DNA and more recently phone records. But the\xa0Moscow PD themselves have remained tight-lipped — until now!\n\nKohberger was [[ extrad]] ited to Idaho on Wednesday night, so he could finally be presented with the probable'`
  - rank 1 (a -0.569): `'her instructions, I began to refer to him as my brother.\n\npuppy with tongue out\n\nHe stayed with us while she traveled. He\xa0is a one-year-old Australian doodle, a [[ breed]]  of dogs that are basically destined to be emotional support.\n\nWhat we know about dogs'`
  - rank 2 (a -0.553): `'ABLE PROVINCIA (\r\n    id                      SERIAL              PRIMARY KEY,\r\n    descripcion             VARCHAR(60)         NOT NULL\r\n);\r\n\r\nCREATE TABLE MAR [[CA]]  (\r\n    id                      SERIAL              PRIMARY KEY,\r\n    descripcion             VARCHAR(30)'`
- L06N2185+ (max|a| 0.5002, A on the exemplars 0):
  - rank 0 (a 0.5): `'Hostels in Tokyo and Kyoto\nAugust 25, 2014 10:18 PM \xa0 [[ Subscribe]] \n\nWhat are some good hostels in Tokyo and Kyoto for a solo traveler?\n\n'`
  - rank 1 (a 0.416): `' week.\nhttp://windsofwinter.org/leptitox-reviews/\n\nEksjö Frisbeegolf - Frisbee\nc/o Johnny Nilsson, Fridkullavägen 1\n57534\xa0Eksjö\n\nTel: [saknas] Information\nE [[-post]] : This is a mailto link\n\nSe all info<|endoftext|>'`
  - rank 2 (a 0.408): `'mments | ask | show | jobs | submit login\nSmall guide to making nice tables [pdf] (ethz.ch)\n167 points by murkle on June 11, 2019 | hide | past | favorite | 43\xa0 [[comments]] \n\nTufte has a lot to say about this (and many viz things'`

## L16N1206+:c1 — Source code and SQL, mid-expression tokens

Size 9, born T_14 (step 1,000,000); scored at T_13 (step 560,000); p_late 0.0045; main False.

- Carrier at T_13: median a+ over the exemplars 0.5928 against q999 0.7079 (ratio 0.837); percentile bracket [0.99, 0.999); max|a| 3.437.
- Carrier at T_14: median 1.33, q999 0.5012, ratio 2.653, present True.
- A_own (exemplar positions in the carrier's own T_13 top-50): 1; rank under A among the layer's 20 grid poles: 1; rank among 22,016: PENDING: needs every pole's top-50 at T_(k-1), S3 results/neurons/w3/grid/1b/out/t13_step560000/topk/layer_16.npz. Scored: PENDING.
- Top-10 lists under A and B: PENDING: job m2_score writes out/m2/samples/raw_examples_freeze.jsonl.
- Pseudo-birth twin: L16N7733- ranks [6, 11, 14, 18, 22, 26, 31, 41, 46], ratio at T_13 0.892, A_own 0

Exemplars (T_15 contexts; a+ at T_(k-1) and T_k):

- rank 1 (a+ 1.6 → 0.971): `'\n/* Based on Java program below published by Lancs University\n *\tClass to take a numbered file after it has been stemmed and sort\n *\tit into pseudo groups where [[ identical]]  stems are adjacent\n *\tRob Hooper\n *\t02/11/'`
- rank 4 (a+ 3.02 → 1.53): `'\tindel_length[length] += 1\r\n\t\t\t\telse:\r\n\t\t\t\t\tindel_length[length] = 1\r\n\t\treturn indel_length\r\n\r\n\r\n\tdef stat_func(self, vcf):\r\n\t\t"""stat the num of snp or indel m [[uation]]  that functional\r\n\t\treturn a dict:\r\n\t\tfunc = {\r\n\t\t\t\'inter'`
- rank 15 (a+ 0.593 → 0.0358): `'           ulong ret = 0;\n            for (int _i = 0; _i < i; ret += CCalculator.arr[_i++]) ;\n            return ret;\n        }\n    }\n    abstract class IModa\n [[   ]]  {\n        protected List<int> moda = new List<int>();\n        ulong max ='`
- rank 16 (a+ 0.339 → 1.33): `'lassroom\ngroup by teachers.last, teachers.first\nhaving count(*) >= 7 and count(*) <= 8\norder by teachers.last;\n\n\nUSE `STUDENTS`;\n-- STUDENTS-2\n-- For each grade [[,]]  report the number of classrooms in which it is taught and the total number of students'`
- rank 20 (a+ 0.0343 → 0): `"ame_from_csv('labs.csv')\r\n    labs_df = drop_missing_values_in_dataframe(labs_df)\r\n    list_of_labs = labs_df['lab_name'].to_list()\r\n    for lab_name in list_of [[_l]] abs:\r\n        df = tests_dataframe.loc[tests_dataframe['lab_name'] =="`
- rank 29 (a+ 0.0203 → 1.72): `'>(trialsUsersCorrectPassword);\n                //originalMalicious.Sort((a, b) => -a.CompareTo(b, conditionNumber));\n                //originalBenign.Sort((a, b [[)]]  => -a.CompareTo(b, conditionNumber));\n                //Queue<Trial> malicious'`
- rank 36 (a+ 1.73 → 1.36): `'nrolled > 20000\nand Gr > 0\ngroup by Campus\norder by Campus\n;\n\n\nUSE `MARATHON`;\n-- MARATHON-1\n-- For each gender/age group, report total number of runners in the [[ group]] , the overall place of the best runner in the group and the overall place of'`
- rank 37 (a+ 0.643 → 1.52): `'= yield Axios.get(\'/api/admin/users\');\n    yield put({ type: \'GET_USER_ADMIN\', payload: elementResponse.data })\n}\n// GET to DB to get all Gender "other" and add [[ them]]  to reduxStore\nfunction* fetchOther() {\n    const elementResponse = yield'`
- … 1 more in the JSON file

Carrier's own top-5 at T_13:

- rank 0 (a 3.44): `'e mapping the palette to line color, not fill, so we had to make a few tweaks. # Importantly, it is now easy to see that "shared room" listings run cheaper more [[ often]]  than other room types, and almost all the pricey rooms are "entire home'`
- rank 1 (a 3.05): `'ang, Ting Jiang, Weihong Zhang, Chunyu Li, Jun Chen, Dandan Xiang, Kejiang Cao, Lian-Wen Qi, Ping Li, Wei Zhu, Wensen Chen, Yan Chen\nThe study was undertaken to [[ describe]]  the profile of patients and the characteristics of all multidrug-resistant gram-negative bacteria ('`
- rank 2 (a 3.02): `'\tindel_length[length] += 1\r\n\t\t\t\telse:\r\n\t\t\t\t\tindel_length[length] = 1\r\n\t\treturn indel_length\r\n\r\n\r\n\tdef stat_func(self, vcf):\r\n\t\t"""stat the num of snp or indel m [[uation]]  that functional\r\n\t\treturn a dict:\r\n\t\tfunc = {\r\n\t\t\t\'inter'`
- rank 3 (a 2.89): `'71 sq. ft. to 3,044,587 sq. ft. The value of building permits is expected to increase by 40.5% from $151,050,592 to $212,190,654. The table below shows the Top  [[10]]  Permits by valuation during October 2023.\n\n* The Top 10 Per'`
- rank 4 (a 2.83): `'e trying to search for having an affair, a casual sex or connection or any other comparable type of raw thing.\n\nIf we were to take the user base and split it up [[ by]]  the nations that the people are located in primary is the united states second? Is'`

Matched random-pole reading (three other grid poles of the layer nearest in log max|a| at T_13), read beside the carrier:

- L16N10389- (max|a| 3.761, A on the exemplars 0):
  - rank 0 (a -3.76): `'uietly asked.\n\nEither his counterpart had shared too much, or McCoy had gotten some medical texts from the Healers that he never wanted to see, Spock decided. “ [[I]]  have been able to determine that I am physically mature and my symptoms matched the profile'`
  - rank 1 (a -2.98): `'launch the CKayTV APK on your Android Smartphone or Tablet and vpn client meraki then sign up to watch Live TV channels. Select the. When the pop-up shows, when [[ its]]  done, youve started streaming your favorite channels without any issues. The installation process'`
  - rank 2 (a -2.86): `'ndent*, and some matters he states just as plainly; and he will be justified by the unprejudiced reader.\n\nOne of the most notable studies of the Jewish Question [[ has]]  come out of the University of the South, at Sewanee, Tennessee.'`
- L16N6982- (max|a| 3.79, A on the exemplars 0):
  - rank 0 (a -3.79): `'tion. Since we are classifying each image as one of the digits from 0 through 9, we only need 10 neurons in this layer. Each of these 10 neurons is connected to [[ all]]  1,024 of the neurons in the previous layer.\n\nNow there’s a'`
  - rank 1 (a -3.33): `'ication. Since we are classifying each image as one of the digits from 0 through 9, we only need 10 neurons in this layer. Each of these 10 neurons is connected [[ to]]  all 1,024 of the neurons in the previous layer.\n\nNow there’s'`
  - rank 2 (a -3.22): `' tf.layers.dense (dense is another name for fully connected) and tell it to create 1,024 neurons. Each of those neurons will be connected to every neuron in the [[ previous]]  layer, which has over 3,000 neurons. This is why it’s'`
- L16N864+ (max|a| 2.941, A on the exemplars 0):
  - rank 0 (a 2.94): `'ant him to cum in your mouth, you incestuous bitch!!" she screamed.\n\nMy mom just kissed Asha hard.\n\n"First, I am going to taste it. I haven\'t tasted it for ages [[,"]]  she said, getting down on her knees. I watched in awe as my mom'`
  - rank 1 (a 2.64): `'have you inside me, that\'s what I miss the most."\n\nYou waited a moment before moving me onto my back and straddling me. "God, I\'ve missed being with you so much [[."]]  you said as you moved your face down to my neck. You traced your warm'`
  - rank 2 (a 2.63): `'m to cum in your mouth, you incestuous bitch!!" she screamed.\n\nMy mom just kissed Asha hard.\n\n"First, I am going to taste it. I haven\'t tasted it for ages," she [[ said]] , getting down on her knees. I watched in awe as my mom, took'`

## L20N5782+:c1 — responsibility and consequences words themselves

Size 15, born T_12 (step 320,000); scored at T_11 (step 180,000); p_late 0.0265; main False.

- Carrier at T_11: median a+ over the exemplars 3.419 against q999 3.758 (ratio 0.910); percentile bracket [0.99, 0.999); max|a| 19.46.
- Carrier at T_12: median 6.474, q999 2.371, ratio 2.730, present True.
- A_own (exemplar positions in the carrier's own T_11 top-50): 1; rank under A among the layer's 20 grid poles: 1; rank among 22,016: PENDING: needs every pole's top-50 at T_(k-1), S3 results/neurons/w3/grid/1b/out/t11_step180000/topk/layer_20.npz. Scored: PENDING.
- Top-10 lists under A and B: PENDING: job m2_score writes out/m2/samples/raw_examples_freeze.jsonl.
- Pseudo-birth twin: L20N8272+ ranks [5, 8, 9, 12, 17, 18, 19, 20, 23, 24, 31, 33, 41, 46, 49], ratio at T_11 0.819, A_own 3

Exemplars (T_15 contexts; a+ at T_(k-1) and T_k):

- rank 2 (a+ 0.873 → 6.75): `'s while trying to save her 9-year-old neighbor, and Mavy Stoddard, whose husband died shielding her from bullets, plan to address the court.\n\n"He has to pay the [[ consequences]]  for what he did, and justice will be served," Hileman said.\n\n'`
- rank 3 (a+ 12.8 → 15.6): `'his application and in his interview. Regardless that the lie could be considered minor (HIPAA violation), it is still wrong. We all make mistakes and should be [[ accountable]]  when asked about them. Ethically, I had to do the right thing even'`
- rank 7 (a+ 11.6 → 7.88): `'sy; it can be unpleasant and frankly inconvenient. It can put you in a stressful situation, forcing yourself, those you love, and colleagues you work with to be [[ accountable]] . However, doing the right thing is always the right thing to do.\n\nProvide'`
- rank 12 (a+ 5.53 → 8.91): `'been prosecuted. Bohm,\xa0herself, says she wants to see Groening declared guilty, but rather than having him sent to jail at his age 93,\xa0feels that he must accept [[ responsibility]]  over his complicity, rather than just claim he was a “cog in'`
- rank 16 (a+ 6.03 → 8.69): `'ble as an adult, while not always fun, does carry a significant amount of power. We begin to control our own destinies, make choices, and learn to deal with the [[ consequences]] —good or bad—of those actions. More important, the responsibilities that we'`
- rank 23 (a+ 2.44 → 6.38): `'losses alone would have been apalling considering how many airplanes, ships and armored vehicles the Alone have destroyed. And yet she never faces any real cons [[quences]]  for her actions.\n    • The fact that Showing Rei compassion was a critical part'`
- rank 26 (a+ 3.51 → 6.73): `'tition is to let everyone know how people feel about the teen years of a teenager. Teens should be free to do what they please...although there are some consequ [[enses]] .\n\n45. "Happy Birthday to You....." 24th B-day for'`
- rank 29 (a+ 2.12 → 4.33): `'ave done so much more damage to the dinar community, than good….I can assure you of this.\n\nMy opinion is that parliament members will very soon have to face the [[ consequences]]  of not addressing these laws and bringing them to a legitimate voting process. Thus their'`
- … 7 more in the JSON file

Carrier's own top-5 at T_11:

- rank 0 (a 19.5): `'t.” Castiel smoothly pulls Anna’s essay out from under her hands. “I’m sure this will be a useful starting point, and any mistakes I make from now on will be my [[ own]] .”\n\n“And…” Anna swallows, looking rather nervous all of a sudden. “'`
- rank 1 (a 17.5): `'what’s happened here.”\n\nSyrian opposition activists reported that helicopters dropped chlorine gas on Kfar Zeita on 11 April, activists said. The government has [[ denied]]  any such attacks.\n\nPsaki said chlorine was not one of the priority one or'`
- rank 2 (a 15.3): `'nd it with your piece:\n\n“I am the sole owner of my quilt square and I choose to donate it to “Piece for Peace”, a branch of WWP.\n\nMy quilt square is entirely my [[ own]]  design: Yes_____ No_____.\nIf yes, I grant WWP unrestricted license'`
- rank 3 (a 14.4): `'enting backend systems that connect and integrate with customer environments. You will work closely with customers, field engineers, and product teams and be an [[ integral]]  part of the thought and design process of our methodologies and practices which guide us in'`
- rank 4 (a 13.8): `"d above (higher-ranking administrators) precipitated Monroe's move. Nelson Perez and Henry Fraind refute that assessment. Both say the change was totally Monroe [['s]]  idea, and that neither their displeasure with her nor union bellyaching, had"`

Matched random-pole reading (three other grid poles of the layer nearest in log max|a| at T_11), read beside the carrier:

- L20N5759+ (max|a| 19.27, A on the exemplars 0):
  - rank 0 (a 19.3): `'pport the entire language of the project. find someone to take java assignment add any and all help you have provided: Support the type annotations of the javax [[.e]] jtag package references, as used in.java files based on javax.view.\n\nCan'`
  - rank 1 (a 16.3): `'if( window.BO == null ) {\n  window.BO = {};\n}\n\n/**\n * (INTERNAL) Append SVG. Called by only constructor.\n * Added: 2021-04-01\n */\nfunction appendSvg [[(e]] _parent, iconname) {\n  var e_div= document.createElement("div");\n'`
  - rank 2 (a 16.3): `'<devEUI>, instead of creating a thing for every device we have. They claim AWS thing should only be used if you also need to publish a state back to the device, [[ e]] .g. when downlink is required.\n\nAWS and AWS IoT integration\nAmazon AWS'`
- L20N5759- (max|a| 18.81, A on the exemplars 0):
  - rank 0 (a -18.8): `' the Design Museum, London.\n\nRecollect by Tan Wei Xiang | Photography by Tim Robinson\n\nMentored by design entrepreneurs such as Tomoko Azumi, Nathan Yong, Maria [[ Jeg]] linska and Adam Markowitz, the designers had to consider the materials’'`
  - rank 1 (a -15.9): `'Is the Vicar in, Pet?\xa0(Heftet)\n\nFrom the Pit to the Pulpit - My Childhood in a Geordie Vicarage\n\n [[Leg]] g i ønskeliste\n\nInnbinding: Heftet\nUtg'`
  - rank 2 (a -15.5): `"2019, 'Jajang Sukmara', 'Belum'),\n('RB44141343', 'Rancabentang', 2020, 'Yoggi Doang', 'Selesai'),\n('RB54541343', 'Rancabentang', 2020, 'Yusman [[ Jeg]] ag', 'Selesai');\n\n-- --------------------------------------------------------\n\n--\n-- Table structure for table `"`
- L20N8203+ (max|a| 18.76, A on the exemplars 0):
  - rank 0 (a 18.8): `'as accurate as each other for the median and the mode the women were much more accurate than the men for the mean.\n\n2. ## mayfeild statistics\n\n48 50 50 50 50 51 [[ ]] 52 52 52 54 54 54 54 54 '`
  - rank 1 (a 18.7): `'\n\n170.81.103.x/24/26266413rewritten\n\n170.81.103.x/24/25266413rewritten\n\n170.81.103.x/24/24266413rewritten\n\n170.81.102.x/24/23266413rewritten\n\n170.81 [[.]] 101.x/24/22266413rewritten\n\n170.81.99'`
  - rank 2 (a 18.3): `'IS further agrees that end-user control and awareness of call blocking being performed on the end-user’s traffic is important.\n\\item \\textsuperscript{26} NOI at [[ ¶]] 39.\n\\item \\textsuperscript{27} As previously noted,'`

## L07N8340+:c4 — mid-word fragments of rare proper names/terms (Templi, Epidemiology, Desulfovibrio, clerical)

Size 5, born T_14 (step 1,000,000); scored at T_13 (step 560,000); p_late 0.0240; main False.

- Carrier at T_13: median a+ over the exemplars 0.08029 against q999 0.2113 (ratio 0.380); percentile bracket below 0.99; max|a| 1.017.
- Carrier at T_14: median 0.2397, q999 0.1334, ratio 1.798, present True.
- A_own (exemplar positions in the carrier's own T_13 top-50): 0; rank under A among the layer's 20 grid poles: 20; rank among 22,016: 22016. Scored: wrong under A (rank 22,016 for certain).
- Top-10 lists under A and B: PENDING: job m2_score writes out/m2/samples/raw_examples_freeze.jsonl.
- Pseudo-birth twin: L07N7004+ ranks [2, 4, 11, 19, 30], ratio at T_13 0.780, A_own 0

Exemplars (T_15 contexts; a+ at T_(k-1) and T_k):

- rank 7 (a+ 0.0766 → 0.436): `', Loeb M, Maccannell T, Malani PN, Mody L, Mylotte JM, Nicolle LE, Roghmann MC, Schweon SJ, Simor AE, Smith PW, Stevenson KB, Bradley SF, Society for Healthcare [[ Epidemi]] ology Long-Term Care Special Interest Group. Surveillance definitions of infections in long-term'`
- rank 8 (a+ 0.0803 → 0.101): `'r Crowley, who called himself "the Great Beast," was a showman, drug fiend, and Satanist of the early twentieth century. He attracted many followers to the Ordo [[ Templ]] i Orientis, seduced (and abused) scores of women, and claimed'`
- rank 14 (a+ 0.425 → 0.24): `'s in the Cloyne Report and in formal Irish government observations about it that accused the Vatican of undermining “the Irish church’s own efforts to deal with [[ cler]] ical child sexual abuse.”\n\nThe text said:\n\nAt the outset, the Holy See'`
- rank 15 (a+ 0.258 → 0.322): `'Changes in metabolic pathways of Desul [[fov]] ibrio alaskensis G20 cells induced by molybdate excess\n\n'`
- rank 47 (a+ 0.0258 → 0.139): `'bined these discoveries with secrets taken from Masonry (and thus the Templars). Spearheaded by orders including the Theosophists, the Golden Dawn, and the Ordo [[ Templ]] i Orientis, occultism rose to a crescendo in the modern era.\n\n'`

Carrier's own top-5 at T_13:

- rank 0 (a 1.02): `'ical elements as well as the arrangement varies from plant to plant based upon numerous wastewater streams produced. Zero Liquid Discharge System Layout. A Zero [[ Liquid]]  Discharge system normally has a Dissipation systems usually are more capital as well as'`
- rank 1 (a 0.973): `' a very customizable service. The technical elements as well as the arrangement varies from plant to plant based upon numerous wastewater streams produced. Zero [[ Liquid]]  Discharge System Layout. A Zero Liquid Discharge system normally has a Dissipation'`
- rank 2 (a 0.914): `' Play Recycling Model are:\n\n  • Ready to use equipment\xa0\n  • No initial CAPEX required\xa0\n  • Quick Mobilization\xa0\n  • Pay per use model\xa0\n  • Rental options on Zero [[ Liquid]]  Discharge\xa0\n  • Flexible renting solutions along with short-term rental options\n\nThis'`
- rank 3 (a 0.845): `' morbidity, disease and physical corruption.5\n\nNurgle overlooking his army\n\nOf the four Gods of Chaos, Nurgle is said to be the most involved with the plight of [[ mort]] als. Those afflicted by his contagions often turn to him in order to escape'`
- rank 4 (a 0.842): `'itis. Acta Orthop 2010;81:199-205. [Crossref] [PubMed]\n  42. Borus T, Brilhault J, Confalonieri N, et al. Patellofemoral joint replacement, an evolving concept. [[ Knee]]  2014;21 Suppl 1:S47-50. [Cross'`

Matched random-pole reading (three other grid poles of the layer nearest in log max|a| at T_13), read beside the carrier:

- L07N4017+ (max|a| 1.021, A on the exemplars 0):
  - rank 0 (a 1.02): `'< Back to Article list\n\nUntitled 1\n\nForgotten your details? Enter email address in Username box and click Reminder. Your details will be emailed to you.\n\nLatest [[ Poll]] \nWhich star is the best at shouting?\nArnold Schwarzenegger\n'`
  - rank 1 (a 0.893): `"a salad?? Is that strange?\n\nSo, I love Doughboys but, on one recent visit I tried the Grandma Rose's Special. It's a Huge bowl of quite excellent chicken noodle [[ soup]]  accompanied by a tuna melt which they describe on their menu as follows: grandma rose"`
  - rank 2 (a 0.811): `"o withstand extreme arctic conditions. This product is for you if you're building in such areas.\n\nYears of experience\n\nRotopile Offices\n\ndesign engineers\n\nLet's [[ Talk]]  About Your\nNext Project.\n\nWe’ll be glad to answer any question! Feel"`
- L07N5613- (max|a| 1.15, A on the exemplars 0):
  - rank 0 (a -1.15): `' assessment in heritage regulation 460\nSzántó, A. 37\nTaalas, M. 482, 483, 486\nTabellini, G. 32, 33\nTalisman, B. 116\nTanner, S. 258\nTapper, R [[.]]  173\ntax incentives 8, 10, 12\ncultural'`
  - rank 1 (a -1.15): `'Bryant, R [[.]]  (2015, February). College Preparation for African American Students: Gaps in'`
  - rank 2 (a -1.09): `'1\nFiling dateSep 25, 1969\nPriority dateSep 25, 1969\nPublication numberUS 3602221 A, US 3602221A, US-A-3602221, US3602221 A, US3602221A\nInventorsBleicken [[ Eric]]  V\nOriginal AssigneeBleicken Eric V\nExport CitationBiBTeX'`
- L07N2194+ (max|a| 1.16, A on the exemplars 0):
  - rank 0 (a 1.16): `'lm’s Koppi Roasters to design elements featuring clean lines and soothing tones. This Hamburg cafe takes the concept even further by serving sweet treats like k [[anel]] buallar cinnamon pastries ideal for fika, Sweden’s traditional coffee break'`
  - rank 1 (a 1.02): `'\nUsta Don [[er]]  is a Turkish restaurant located in Mall of Qatar that offers a wide variety of delicious'`
  - rank 2 (a 0.998): `' your response. Here are the devices present in my system: CPU - i5 4300M 2.6GHz quad core IGP - HD4600 15" 1920x1080 2x2 11b/g/n Wileless LAN M.2 Adapter Intel [[ ®]]  Ethernet Connection 1217-LM Realtek Blutooth 4.0'`

## L10N9052+:c2 — NPR 'Transcript provided by NPR, Copyright'

Size 5, born T_14 (step 1,000,000); scored at T_13 (step 560,000); p_late 0.0450; main False.

- Carrier at T_13: median a+ over the exemplars 0.0235 against q999 0.335 (ratio 0.070); percentile bracket below 0.99; max|a| 1.724.
- Carrier at T_14: median 0.3077, q999 0.2113, ratio 1.456, present True.
- A_own (exemplar positions in the carrier's own T_13 top-50): 0; rank under A among the layer's 20 grid poles: 20; rank among 22,016: 22016. Scored: wrong under A (rank 22,016 for certain).
- Top-10 lists under A and B: PENDING: job m2_score writes out/m2/samples/raw_examples_freeze.jsonl.
- Pseudo-birth twin: L10N9100+ ranks [11, 16, 21, 33, 37], ratio at T_13 0.778, A_own 0

Exemplars (T_15 contexts; a+ at T_(k-1) and T_k):

- rank 4 (a+ 0.123 → 0.528): `'s a new album out, "AudioLust & HigherLove." Thank you so much for joining us.\n\nLEWIS: Thank you so much. Thanks for the great chat. Transcript provided by NPR, [[ Copyright]]  NPR.\n\nAyesha Rascoe is a White House correspondent for NPR. She'`
- rank 9 (a+ 0.0235 → 0.302): `"artment. He joined me from the studios at Berkeley School of Journalism. Thanks so much for talking with us.\n\nHAY: Thank you, ma'am. Transcript provided by NPR, [[ Copyright]]  NPR.\n\nMore On This Topic<|endoftext|>"`
- rank 12 (a+ 0 → 0.308): `' in Austria. Vesta, thank you for speaking with us.\n\nELEUTERI: Thank you so much for having me.\n\n(SOUNDBITE OF FETE\'S "THE ISLANDS") Transcript provided by NPR, [[ Copyright]]  NPR.\n\nNPR transcripts are created on a rush deadline by an NPR contractor.'`
- rank 19 (a+ 0 → 0.201): `'or ESPN.com and ESPN, the magazine, talking about the phenomenon of Tim Tebow. You can also hear Howard on WEEKEND EDITION SATURDAY. Transcript provided by NPR, [[ Copyright]]  NPR.<|endoftext|>'`
- rank 49 (a+ 0.0444 → 0.344): `'hile, this turf war is blocking the bureau from doing much to help.\n\nChris Arnold, NPR News.\n\n(SOUNDBITE OF I/O\'S "BAIT AND SWITCH") Transcript provided by NPR, [[ Copyright]]  NPR.\n\nNPR correspondent Chris Arnold is based in Boston. His reports are heard'`

Carrier's own top-5 at T_13:

- rank 0 (a 1.72): `"idea to think about why people use them all the time, but then twist them around so it doesn't sound like a cliche, but the sentiment is still there.”\n\nSpeaking [[ of]]  cliches: Near the end of this month, Brauns' life will come"`
- rank 1 (a 1.68): `'ys on the highway, and sent a geyser shooting high into the air.\n\nWe’re working on getting more information on the cause and impact of this water main break.\n\n( [[Submitted]]  photo)\n\n(Photo: Cydny Neville)\n\nSend news and photos to Potom'`
- rank 2 (a 1.67): `'t, glasses pushed down on her nose. She is one of six police officers assigned to the special investigations unit now headed by Vivian Monroe. Clark regards the [[ reporter]]  who has identified himself as a staff writer for New Times.\n\n"Oh," she says'`
- rank 3 (a 1.66): `'d the soldier to death ‘in order to advance your extremist cause’, as if trying to decapitate a man in a London street was not crime enough?\n\nAs Brendan O’Neill [[ pointed]]  out on spiked amid the immediate panicky political and media response to the killing,'`
- rank 4 (a 1.62): `'hat’s been made available to them.”\n\nBaker praised the T for its plan to hire the new capital workers to help improve the system.\n\nMatt Stout of the Globe staff [[ contributed]]  to this report.<|endoftext|>'`

Matched random-pole reading (three other grid poles of the layer nearest in log max|a| at T_13), read beside the carrier:

- L10N4041- (max|a| 1.711, A on the exemplars 0):
  - rank 0 (a -1.71): `'| for each package.\n    """\n    package_to_images = defaultdict(list)\n    for image in images:\n        package_to_images[image[\'package\']].append(image)\n\n    to [[_gc]]  = []\n    for _, images in package_to_images.items():\n        # Because the'`
  - rank 1 (a -1.57): `"                 artifact_repository_url, '--format=json', '--quiet'])\n    if not ok:\n        return None\n    return list(json.loads(text))\n\n\ndef find_images_to [[_gc]] (images: List[Dict[str, Any]], num_last_to_keep) -> List"`
  - rank 2 (a -1.45): `'e m x n numerical data matrix A, where m is the number of rows and n is the number of features\n• Decompose it into 3 matrices as shown here:\n• Choose k singular [[ values]]  based on the diagonal matrix and truncate (trim) the 3 matrices accordingly:\n\n'`
- L10N9794- (max|a| 1.947, A on the exemplars 0):
  - rank 0 (a -1.95): `'y calling stop.\n# This can be fiddlier than you might think, because if an exception is raised in the setUp then tearDown is not called.\n# unittest.TestCase.add [[Cleanup]] () makes this easier:\n# \n\nc_ MyTest(T..\n        ___ setUp'`
  - rank 1 (a -1.78): `'ion source code. APR pools offer some extensibility mechanisms, like the ability to have custom “user data” attached to the pool, and mechanisms for registering [[ cleanup]]  functions that get called when the pool is destroyed. Subversion makes use of these'`
  - rank 2 (a -1.75): `"# Define your item pipelines here\n#\n# Don't forget to add your pipeline to the ITEM_PIPELINES setting\n# See: http://doc.scrapy.org/en/0.16/topics/item-p [[ipeline]] .html\nfrom scrapy.exceptions import DropItem\nimport hashlib\nimport bleach\n\n\nclass"`
- L10N3963- (max|a| 1.494, A on the exemplars 0):
  - rank 0 (a -1.49): `' real and surrogate (shuffled) data, for all nonself channel pairs (connections), for each participant dyad, for each gaze condition, and in Theta and Alpha EEG [[ bands]]  (Fig. 1 C and D). In the subsequent network diagrams (F'`
  - rank 1 (a -1.38): `' Polo S, Monaco MC, Verani A, et al. Nonproductive human immunodeficiency virus type 1 infection of human fetal astrocytes: independence from CD4 and major chem [[ok]] ine receptors. Virology 1999; 264:370–384'`
  - rank 2 (a -1.31): `'Dendritic cell-derived interleuk [[in]] -15 is crucial for therapeutic cancer vaccine potency\n\nOncoimmunology.'`

## L10N9794+:c2 — get/delete/add/list verb inside identifiers, CLI args and function names

Size 12, born T_10 (step 100,000); scored at T_9 (step 60,000); p_late 0.0280; main False.

- Carrier at T_9: median a+ over the exemplars 3.556 against q999 4.217 (ratio 0.843); percentile bracket [0.99, 0.999); max|a| 34.36.
- Carrier at T_10: median 3.732, q999 1.334, ratio 2.799, present True.
- A_own (exemplar positions in the carrier's own T_9 top-50): 0; rank under A among the layer's 20 grid poles: 20; rank among 22,016: 22016. Scored: wrong under A (rank 22,016 for certain).
- Top-10 lists under A and B: PENDING: job m2_score writes out/m2/samples/raw_examples_freeze.jsonl.
- Pseudo-birth twin: L10N9052- ranks [1, 3, 6, 12, 13, 28, 32, 34, 35, 36, 46, 48], ratio at T_9 0.966, A_own 0

Exemplars (T_15 contexts; a+ at T_(k-1) and T_k):

- rank 5 (a+ 6.57 → 5.18): `'sync function getAutorByAutorId (autorId) {\n  return await autorRepository.getAutorByAutorId(autorId)\n}\n\nexport default { createAutor, updateAutor, deleteAutor, [[ get]] Autores, getAutorByAutorId }\n<|endoftext|>'`
- rank 7 (a+ 2.45 → 4.84): `"const db = require('./db');\nconst helper = require('../helper');\nconst config = require('../config');\n\n\n\n\n// [[ get]]  top 10 messages\nasync function getMultiple(page = 1){\n  const"`
- rank 14 (a+ 3.52 → 5.09): `"op, schema);\n    const { error } = result;\n    if (error) {\n      const {\n        details: [{ message }]\n      } = error;\n      return message;\n    }\n  };\n\n  // [[ get]]  methods\n\n  couseRouter.get('/', (req, res) => {\n   "`
- rank 16 (a+ 12.5 → 8.19): `'utores()\n}\n\nasync function getAutorByAutorId (autorId) {\n  return await autorRepository.getAutorByAutorId(autorId)\n}\n\nexport default { createAutor, updateAutor, [[ delete]] Autor, getAutores, getAutorByAutorId }\n<|endoftext|>'`
- rank 18 (a+ 1.91 → 0.951): `" = this.category(arg);\n        this.action = this.action(arg);\n    }\n\n\n    action(arg = {}) {\n        if(arg.list || arg.l) {return 'list';}\n        else if(arg [[.add]]  || arg.a) {return 'add';}\n        // add delete func\n"`
- rank 20 (a+ 9.33 → 4.94): `'re("express");\nconst app = express();\nconst cors = require(\'cors\');\nconst path =require("path");\nconst{uploadImage, updateContact, createContact, deleteContact, [[ getAll]] Contacts} = require("./controller/contactController");\n\n\n// Middleware\napp.use(express.json'`
- rank 23 (a+ 7.85 → 3.49): `"tion(arg = {}) {\n        if(arg.list || arg.l) {return 'list';}\n        else if(arg.add || arg.a) {return 'add';}\n        // add delete func\n        else if(arg [[.delete]]  || arg.d) {return 'delete';}\n        else {return 'Enter"`
- rank 26 (a+ 4.73 → 3.29): `"(arg.list || arg.l) {return 'list';}\n        else if(arg.add || arg.a) {return 'add';}\n        // add delete func\n        else if(arg.delete || arg.d) {return ' [[delete]] ';}\n        else {return 'Enter a command';}\n    }\n    note"`
- … 4 more in the JSON file

Carrier's own top-5 at T_9:

- rank 0 (a 34.4): `'of at least a portion of a built-in outdoor kitchen that is substantially constructed of the outdoor building material.\n\n[0008] The invention further includes a [[ method]]  of providing a weather-resistant outdoor kitchen. The method includes making available a plurality of'`
- rank 1 (a 33.4): `'c(id);\n\n            if (enrollment == null)\n            {\n                return NotFound();\n            }\n\n            return enrollment;\n        }\n\n        // [[ PUT]] : api/EnrollmentsAPI/5\n        // To protect from overposting'`
- rank 2 (a 32.7): `'tChannelList(plan);\n            if (plan == null)\n            {\n                return NotFound();\n            }\n\n            return plan;\n        }\n\n        // [[ PUT]] : api/Plans/5\n        // To protect from overposting attacks, please'`
- rank 3 (a 31.5): `'    if (personalvisita == null)\r\n            {\r\n                return NotFound();\r\n            }\r\n\r\n            return personalvisita;\r\n        }\r\n\r\n        // [[ PUT]] : api/Personalvisita/5\r\n        [HttpPut("{id}'`
- rank 4 (a 29.2): `" | 'nil'.\niterator --> [tree|iterator] | [].\nThe (sub)trees in an iterator are there for the sake of the key\nand _one_ of the subtrees.\n\niterator({_Size, Tree}) [[ ->\n]] iterator(Tree, []).\n\niterator({_X, nil, _Right} = Tree"`

Matched random-pole reading (three other grid poles of the layer nearest in log max|a| at T_9), read beside the carrier:

- L10N5687+ (max|a| 36.58, A on the exemplars 0):
  - rank 0 (a 36.6): `' please contact the two PC chairs:\nCatalin Hritcu (catalin.hritcu@gmail.com),\nJasmin Christian Blanchette (j.c.blanchette@vu.nl)\n\nRelated Resources\n\nICDMML 2020 [[\xa0\xa0]]  【EI SCOPUS】2020 International Conference on Data Mining and Machine Learning'`
  - rank 1 (a 32.5): `' with the product topology. I think I have an example for when A is not countable. Suppose A is the set of all … Read more\n\n## Landau’s proof that ζ(s)=∏p11−p−s [[\\]] zeta(s)=\\prod_{p} \\frac{1}{1-p'`
  - rank 2 (a 31.2): `'ure 3(b), we consider a C++ component for a low-level real-time implementation. Alternatively, one may want to derive VHDL or Verilog HDL to synthesize an FPGA. [[444]] VHSIC or Verilog hardware description language (VHDL or Verilog'`
- L10N5687- (max|a| 24.03, A on the exemplars 0):
  - rank 0 (a -24): `'ferencesconference list [pretty|summary|count|delim <string>]\n\nlockLock a conference so no new members will be allowed to enterconference <confname> lock\n\nmuteM [[utes]]  a conference memberconference <confname> mute <member_id>|all|last'`
  - rank 1 (a -20.7): `'name> <value>\n\nsetvarsets the value of a confernce variableconference <confname> setvar <var-name> <var-value>conference\xa0testconf setvar\xa0mycustomvar 1234\nstopSt [[ops]]  any queued audio from playing into conferenceconference <confname> [current|all'`
  - rank 2 (a -18.5): `'hing — let me explain. I was in the middle of copying some files to a new, shared directory that I setup to share with my LAN. This is when I had to ‘chown’ (CH [[ange]]  OWNership) the copied files — I started to type the command\n\nsudo'`
- L10N10529+ (max|a| 51.14, A on the exemplars 0):
  - rank 0 (a 51.1): `' using non-profit covers ect. No different than the rich Americans try to do. Waltons, Koch Bros. Buffett ect.\n\nP.S. You know things are getting bad in the U.S. [[ when]]  a corporation like Berger King is looking to Canada to relocate. Thank you BO.'`
  - rank 1 (a 47.5): `"m a different dimension.[2] Despite this, it seems that he has at least spent time or somehow lived in Blaze's home dimension prior to the events of Sonic Rush, [[ as]]  she expresses familiarity with him.\n  • Curiously, in both Sonic Rivals"`
  - rank 2 (a 42.3): `' and which may have been mentioned in the last portion of our inscription. That the final restorer took some pains to follow his copy with care is proved by the [[ fact]]  that he corrected the erroneous\xa0u to\xa0o in navebos, and this'`

## L19N7374-:c1 — web boilerplate: Skip to Content, You are here, http

Size 10, born T_13 (step 560,000); scored at T_12 (step 320,000); p_late 0.0000; main False.

- Carrier at T_12: median a+ over the exemplars 1.957 against q999 2.371 (ratio 0.825); percentile bracket [0.99, 0.999); max|a| 15.85.
- Carrier at T_13: median 4.18, q999 1.585, ratio 2.637, present True.
- A_own (exemplar positions in the carrier's own T_12 top-50): 0; rank under A among the layer's 20 grid poles: 20; rank among 22,016: 22016. Scored: wrong under A (rank 22,016 for certain).
- Top-10 lists under A and B: PENDING: job m2_score writes out/m2/samples/raw_examples_freeze.jsonl.
- Pseudo-birth twin: L19N2718+ ranks [2, 4, 9, 11, 22, 30, 39, 40, 44, 49], ratio at T_12 0.869, A_own 2

Exemplars (T_15 contexts; a+ at T_(k-1) and T_k):

- rank 2 (a+ 6.48 → 6.06): `'Fairfield Ledger\n [[http]] ://fairfield-ia.villagesoup.com/p/1060541\n\n'`
- rank 5 (a+ 3.99 → 5.53): `'You are [[ here]] \n\nDoepfer Dark Time\n\nStep Sequencer\nPublished August 2013'`
- rank 7 (a+ 0.527 → 3.61): `'Skip to [[ Content]] \nDark Mode\n\nBest Wondrous Physick Combinations in Elden Ring\n\n'`
- rank 12 (a+ 2.87 → 3.98): `'You are [[ here]] \n\nIN AND AROUND CAR ACTIVITIES (Education only - no car seat'`
- rank 16 (a+ 0.896 → 3.91): `'Skip to [[ Content]] \n\nRWBY: Amity Arena Launches Today on iOS and Android\n\nThe'`
- rank 20 (a+ 4.17 → 5.27): `'You are [[ here]] \n\nThis ATEM (mini) Tips” series\xa0of short video tutorials for'`
- rank 21 (a+ 1.05 → 4.38): `'Skip to [[ Content]] \n\nWoW Insider has the latest on the Mists of Pandaria!\n\nTot'`
- rank 28 (a+ 0.772 → 3.88): `'Skip to [[ Content]] \n\nWoW Insider has the latest on the Mists of Pandaria!\n\nThe'`
- … 2 more in the JSON file

Carrier's own top-5 at T_12:

- rank 0 (a -15.8): `'gs, this innovative system caters to diverse cooking needs with ease.\n\nXL capacity\n\nOur Score\n\nExperience the ultimate culinary versatility with the Ninja DG551 [[ Food]] i Smart XL 6-in-1 Indoor Grill. This sleek appliance combines the'`
- rank 1 (a -14.8): `'maller in size than regular blenders for making sauce.\n\nCheck latest price on Amazon >>\n\n4. Ninja BN801 Professional Plus Kitchen System with Auto-IQ\n\nThe Ninja [[ Food]] i is a great choice for hot sauce, but the Professional Plus Kitchen system has'`
- rank 2 (a -14.1): `'ystem With Auto IQ\n\n5. Instant Ace Nova Cooking Blender\n\nBest Blender For Making Hot Sauce – Detailed Reviews\n\n1. Ninja Foodi Cold And Hot – Top Pick\n\nThe Ninja [[ Food]] i Cold and Hot is a revolutionary new way to cook your meals.\xa0\n\n'`
- rank 3 (a -13.1): `'inimizes smoke production during grilling, allowing users to enjoy a truly smokeless cooking experience right in their kitchen.\n\nSmoke-control system\n\nThe Ninja [[ Food]] i Grill XL features a smoke-control system that helps to minimize smoke production during gr'`
- rank 4 (a -12.9): `'. Ninja BN801 Professional Plus Kitchen System With Auto IQ\n\n5. Instant Ace Nova Cooking Blender\n\nBest Blender For Making Hot Sauce – Detailed Reviews\n\n1. Ninja [[ Food]] i Cold And Hot – Top Pick\n\nThe Ninja Foodi Cold and Hot is'`

Matched random-pole reading (three other grid poles of the layer nearest in log max|a| at T_12), read beside the carrier:

- L19N6189+ (max|a| 16.56, A on the exemplars 0):
  - rank 0 (a 16.6): `"tside of the home or not. Or join the PTA or not. Or send your kid to private school or not. Neither they, nor any other rational human being, believes there is [[ One]]  Moral And Acceptable Way To Live Your Life. They'll do them, you"`
  - rank 1 (a 16.2): `'h as Julian T. Jackson have argued, de Gaulle and his followers were revolutionaries, unlike the Dutch, Belgian, and other governments in exile in London.[17] A [[ third]]  option might be that neither considered that a fully free, legitimate, sovereign, and'`
  - rank 2 (a 14.5): `' through. Some would try their very best to hold on to every second of their childhood, whereas some would just pray for the arrival of adulthood. But no matter [[ which]]  type, they are still helpless when it comes to altering the hour-glass that'`
- L19N1554+ (max|a| 15.06, A on the exemplars 0):
  - rank 0 (a 15.1): `'me account, 2,500 email addresses and more. We offer both software products and customized software development services. To examine the involvement of AIP1Alix [[ in]]  HTLV-1 budding, we q mutant connecting to a sql server asp'`
  - rank 1 (a 14.4): `'Personal tools\n\nA2: Basolateral ciliogenesis, [[ basal]]  progenitor delamination, and the formation of the subventricular zone in developing'`
  - rank 2 (a 14.4): `'comes effective the first of the month following serve date the enrollment form is signed and submitted to Human Resources. The EARP served associates with Rab4 [[-positive]]  endosomes and promotes recycling of internalized transferrin receptor (TFRC)'`
- L19N1154+ (max|a| 13.61, A on the exemplars 0):
  - rank 0 (a 13.6): `'itating noise—from other nearby electronic products and transmitters, including older cordless phones that might use the same frequency bands as your monitor.\n\n [[Think]]  about the size of your home and your daily routine when deciding which brand/model to'`
  - rank 1 (a 13.5): `' keep tabs on your little one(s) during nap time or nighttime, and still go about your day. Generally, the number one benefit provided by the best baby monitors [[ are]]  peace of mind, followed closely by the benefits of security and convenience. Plus,'`
  - rank 2 (a 11.9): `"ciding which brand/model to buy. If you'll be making calls during nap time, for example, look for monitors with lights that let you know when your baby is awake [[.]]  Of course you can accomplish the same thing by turning down the sound on a video"`

## L24N5066-:c1 — commas separating company names in the cable glands market report list

Size 15, born T_12 (step 320,000); scored at T_11 (step 180,000); p_late 0.0130; main False.

- Carrier at T_11: median a+ over the exemplars 6.418 against q999 6.683 (ratio 0.960); percentile bracket [0.99, 0.999); max|a| 43.23.
- Carrier at T_12: median 9.49, q999 4.732, ratio 2.006, present True.
- A_own (exemplar positions in the carrier's own T_11 top-50): 0; rank under A among the layer's 20 grid poles: 20; rank among 22,016: 22016. Scored: wrong under A (rank 22,016 for certain).
- Top-10 lists under A and B: PENDING: job m2_score writes out/m2/samples/raw_examples_freeze.jsonl.
- Pseudo-birth twin: L24N3927- ranks [2, 8, 11, 13, 14, 16, 17, 19, 21, 24, 29, 36, 39, 40, 48], ratio at T_11 0.999, A_own 2

Exemplars (T_15 contexts; a+ at T_(k-1) and T_k):

- rank 1 (a+ 6.16 → 14.4): `'\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\n\nKey Companies Analysis:- Amphenol Industrial Products Group, Caledonian Cables Ltd, CMP Products, Copper Crouse-Hinds (Eaton [[),]]  Cortem Group, Elsewedy Electric, Emerson Industrial Automation, BARTEC'`
- rank 3 (a+ 6.77 → 14.8): `'out this news \xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\n\nKey Companies Analysis:- Amphenol Industrial Products Group, Caledonian Cables Ltd, CMP Products [[,]]  Copper Crouse-Hinds (Eaton), Cortem Group, Elsew'`
- rank 4 (a+ 5.12 → 13.4): `'\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\n\nKey Companies Analysis:- Amphenol Industrial Products Group, Caledonian Cables Ltd, CMP Products, Copper Crouse-Hinds (Eaton), Cortem Group [[,]]  Elsewedy Electric, Emerson Industrial Automation, BARTEC FEAM, BAR'`
- rank 5 (a+ 10.7 → 11.8): `'fore-buying/457120 Make an Inquiry about this news \xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\n\nKey Companies Analysis:- Amphenol Industrial Products Group [[,]]  Caledonian Cables Ltd, CMP Products, Copper Crouse-Hinds'`
- rank 6 (a+ 9.63 → 13.7): `' an Inquiry about this news \xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\n\nKey Companies Analysis:- Amphenol Industrial Products Group, Caledonian Cables Ltd [[,]]  CMP Products, Copper Crouse-Hinds (Eaton), Cortem Group'`
- rank 11 (a+ 5.17 → 8.72): `'ian Cables Ltd, CMP Products, Copper Crouse-Hinds (Eaton), Cortem Group, Elsewedy Electric, Emerson Industrial Automation, BARTEC FEAM, BARTEC Group, Jacob GmbH [[,]]  Metal Craft Industries, R.Stahl AG, Sealcon, TE Connectivity, Hub'`
- rank 13 (a+ 7.15 → 10.4): `'oup, Caledonian Cables Ltd, CMP Products, Copper Crouse-Hinds (Eaton), Cortem Group, Elsewedy Electric, Emerson Industrial Automation, BARTEC FEAM, BARTEC Group [[,]]  Jacob GmbH, Metal Craft Industries, R.Stahl AG, Sealcon, TE'`
- rank 15 (a+ 6.45 → 11): `'al Products Group, Caledonian Cables Ltd, CMP Products, Copper Crouse-Hinds (Eaton), Cortem Group, Elsewedy Electric, Emerson Industrial Automation, BARTEC FEAM [[,]]  BARTEC Group, Jacob GmbH, Metal Craft Industries, R.Stahl AG'`
- … 7 more in the JSON file

Carrier's own top-5 at T_11:

- rank 0 (a -43.2): `'to the columbite supergroup are: the general stoichiometry MO2; the crystal structure based on the hexagonal close packing (hcp) of anions (or close to it); the [[ six]] -fold coordination number of M-type cations (augmented to eight-fold in the'`
- rank 1 (a -29.3): `'p view of the atomic models of the four kinds of Fe-N centers. From the atomic models, one can see that the normal FeN4 and normal FeN3C moieties exhibit a plan [[ar]]  structure in the carbon matrices, while FeN4 SW and FeN3C'`
- rank 2 (a -28.6): `'A wide prototyping area with some extra features:\n  • 1.0 Arduino pinout\n  • 1 Reset button\n  • 1 ICSP connector\n  • 14-pin SMD footprint (50 mils pitch)\n  • 20 [[-pin]]  Through Hole footpront (100 mils pitch)\n\nSchematic & Reference Design'`
- rank 3 (a -26.5): `"  In contrast conda install <package> probably will install all dependencies, including C/C++, FORTRAN, etc., even Haskell libraries, because conda doesn't just [[ manage]]  Python packages like pip, it manages environments. So if you had Anaconda or"`
- rank 4 (a -26.4): `'e if and only if it is observationally equivalent to . As explained in Sect. 3.5, the test wrapper with embedded supervisor (denoted by ) implements an FSM over [[ the]]  same input and output alphabet as . The test executable runs test suite against ; this'`

Matched random-pole reading (three other grid poles of the layer nearest in log max|a| at T_11), read beside the carrier:

- L24N269- (max|a| 42.37, A on the exemplars 0):
  - rank 0 (a -42.4): `'ing email messages are stored by your Internet Service Provider.\n\nMailing list: A single email address that comprises several different email addresses. Example [[:]]  An automated software allows you to send email to one address. Then your message ('`
  - rank 1 (a -33.6): `' email address something like …@gmail.com, …@yahoo.com and so on, use your own domain;\n  • your sender address is the same of the receiver address (do not test\xa0 [[from]]  name@domain.com to name@domain.com many mail service providers drop that emails'`
  - rank 2 (a -32.9): `"ing 8 of Cadence Design Systems' San Jose, Calif. office. Cadence Design Systems, D2S and e-Shuttle are sponsoring the training sessions at no cost to attendees [[.]]  For both locations, the morning session will take place from 10:00 a"`
- L24N1234- (max|a| 42.05, A on the exemplars 0):
  - rank 0 (a -42): `'\nThis article provides information and/or facts that contain spoilers.\nPlease read with caution or refrain from proceeding further.\n\n"Take up arms, newly arisen [[.]]  For my kind do not heed the toothless"\n\n"Dragons are creatures wre'`
  - rank 1 (a -41): `' family, attendees can expect an experience that embodies the vibrant spirit of Patchogue.\n\n“Alive After 5” continues on August 8, with a rain date of August 15 [[.]]  For those planning to attend, parking and shuttle services will be available to accommodate the'`
  - rank 2 (a -35.6): `'ch also ignited a war between alien races and was misused by invaders for their malicious means (such as Alien Bat reviving the original Zetton into Zetton II). [[ Seeing]]  the horrifying results of his actions, Hikari resigned from the Space Science and'`
- L24N2862+ (max|a| 39.85, A on the exemplars 0):
  - rank 0 (a 39.8): `'shyness by putting your resistance in perspective. The importance of your job search, earning a living or performing well in your position far outweigh any real [[ or]]  imagined consequences that could result from these reasons for resisting.\n\nOf course, resistance can'`
  - rank 1 (a 37.5): `'ied, you were justified in the name of the Lord Jesus Christ" (1 Cor. 6:9-11 NIV).\n\n7. Its analysis of human nature.\nThe Bible says that society\'s real problems [[ are]]  problems of the heart. In an age of information and technology, failures of character'`
  - rank 2 (a 36.9): `' of Rome.\n\nThe consolations of Anglo-Catholicism, then, were insufficient for the\nspiritual needs of this scion of the Low Church.\n\nWhat were those needs?\n\nWere [[ they]] , indeed, _spiritual_ needs, as he suggests by the title of'`
