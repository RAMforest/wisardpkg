Tempo de treino e inferência (distributive), todos os modelos de um conjunto na mesma arquitetura (bits/tupla: a do melhor one-shot na validação, a mesma em todas as sementes); b, rho0 e tau do modelo por épocas escolhidos na validação de cada semente; o treino one-shot não depende do bleaching e é medido uma vez por semente, o mesmo nos três modos; média ± dp entre as sementes, acurácias em %.

| conjunto | bits/tupla | modelo | bleaching | sementes | épocas rodadas | melhor época | treino (s) | inferência (s) | inferência (µs/ex.) | acc teste | acc grid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| higgs | 16/8 | One-shot | Iterativo | 0,1,2,3,4 | - | - | 0,2457 ± 0,0972 | 0,9389 ± 0,1191 | 63,84 ± 8,10 | 58,61 ± 0,33 | 58,61 ± 0,33 |
| higgs | 16/8 | One-shot | DAB | 0,1,2,3,4 | - | - | 0,2457 ± 0,0972 | 0,1674 ± 0,0147 | 11,38 ± 1,00 | 62,47 ± 0,62 | 62,47 ± 0,62 |
| higgs | 16/8 | One-shot | rDAB | 0,1,2,3,4 | - | - | 0,2457 ± 0,0972 | 0,2107 ± 0,0401 | 14,32 ± 2,73 | 63,08 ± 0,90 | 63,08 ± 0,90 |
| higgs | 16/8 | Épocas | Iterativo | 0,1,2,3,4 | 18,2 ± 2,7 | 15,8 ± 5,0 | 5,3806 ± 0,8373 | 0,1976 ± 0,0203 | 13,44 ± 1,38 | 67,71 ± 0,48 | 67,71 ± 0,48 |
| higgs | 16/8 | Épocas | DAB | 0,1,2,3,4 | 18,8 ± 2,7 | 16,6 ± 4,4 | 5,1143 ± 0,7197 | 0,1218 ± 0,0112 | 8,28 ± 0,76 | 67,82 ± 0,62 | 67,82 ± 0,62 |
| higgs | 16/8 | Épocas | rDAB | 0,1,2,3,4 | 18,8 ± 2,7 | 17,2 ± 4,7 | 5,0685 ± 0,6640 | 0,1157 ± 0,0100 | 7,87 ± 0,68 | 67,76 ± 0,33 | 67,76 ± 0,33 |
