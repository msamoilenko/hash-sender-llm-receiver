# Resultado — artigo único: até onde vale um emissor só de hash (execução de 2026-11-05/06)

**Status:** uma execução do teste pré-registrado em `919fd75` (`docs/PREREGISTRO_ARTIGO_UNICO.md`), código congelado em `4013e6f`, sem re-run. Reportado qualquer que fosse. Primária e A completas em 2026-11-05; B completa em 2026-11-06 00:35.

## Resposta curta

**Desfavorável ao esquema no endpoint primário.** Para entregar 100% (NACK de 2 B + reserva crua), o emissor só de hash gasta **mais** que o PPM o1 treinado no domínio, e não é equivalente na margem de 1 byte. O PPM o1 usa cerca de 3 vezes a memória do emissor do esquema (24.844 B contra 8.190 B, host arm64). Entre ~8 e ~25 KB de emissor, o esquema é o mais barato medido; a partir do o1, os emissores n-grama do domínio gastam menos bits e menos tempo no ar. A busca maior no receptor (secundária B) inverte o sinal contra o o1 em bits, ao custo de minutos de receptor por falha, e ainda perde em tempo no ar em 4 de 5 enlaces.

## Endpoint primário (pré-registrado)

- Diferença pareada do custo para entregar 100% (reserva crua) contra o PPM o1 no domínio: **18,4 bits por mensagem, IC95 bootstrap [10,9; 26,3]** <!--gate-cost csv=single_analysis.csv ref=ppm_o1 var=raw-->
- Normal (sensibilidade): [10,7; 26,2]. Com +2 bits de terminação no PPM: 16,4 [8,9; 24,3].
- **Leitura pela regra pré-registrada:** sinal — *o esquema gasta mais que o PPM o1* (IC inteiro acima de 0); equivalência — *não equivalente* (IC sai de [−8, +8]).
- Previsão do pré-registro: ponto +2,3, faixa −21,7 a +59,7, leitura esperada "indeterminado". O observado ficou dentro da faixa, mas fora da leitura esperada no ponto: p ficou abaixo do desenvolvimento.

## Componentes

- **Entrega na primeira tentativa:** 857/1000, 140,9 bits do esquema por mensagem em média, p = 85,7% (Wilson 83,4%–87,7%), 0 erradas aceitas <!--gate csv=single_primary.csv-->
- Previsão: p entre 74,4% e 96,5%, ponto 90%. Dentro da faixa, abaixo do ponto.
- **Custo médio para 100%:** 192,8 bits (reserva crua; IC95 [185,0; 201,0]); 178,4 bits com reserva Unishox2 ([172,6; 184,3]).
- **Bits por ponto da fronteira** (overhead de 40 em todos; PPM ideais): cru 344,0; smaz2 238,2; Unishox2 239,3; PPM o1 174,4; o2 podado 16 KiB 258,6; 32 KiB 143,7; 64 KiB 140,5; o2 cheio 140,5; o3 114,6 (`docs/medidas/frontier_test.csv`).

## Fronteira memória × bits (secundário; memória de `docs/medidas/sender_memory.csv`)

| Emissor | Flash (B) | Bits para 100% | Diferença do esquema (custo cru − ponto), IC95 |
|---|---|---|---|
| Texto cru (só etiqueta) | 7.824 | 344,0 | −151,1 [−159,1; −142,9] |
| **Esquema `10:4` + NACK + reserva crua** | **8.190** | **192,8** | — |
| smaz2 | 10.360 | 238,2 | −45,3 [−53,1; −37,4] |
| Unishox2 | 15.697 | 239,3 | −46,4 [−54,2; −38,4] |
| PPM o1 | 24.844 | 174,4 | +18,4 [10,9; 26,3] |
| PPM o2 podado 16 KiB | 25.786 | 258,6 | −65,8 [−74,1; −57,3] |
| PPM o2 podado 32 KiB | 42.168 | 143,7 | +49,2 [41,8; 56,9] |
| PPM o2 cheio | 96.462 | 140,5 | +52,3 [44,8; 60,2] |
| PPM o3 | 440.626 | 114,6 | +78,2 [70,7; 86,0] |

- Contra o smaz2: -45,3 bits [-53,1; -37,4] <!--gate-cost csv=single_analysis.csv ref=smaz2 var=raw-->
- Contra o Unishox2: -46,4 bits [-54,2; -38,4] <!--gate-cost csv=single_analysis.csv ref=unishox2 var=raw-->
- Contra o PPM o2 cheio: 52,3 bits [44,8; 60,2] <!--gate-cost csv=single_analysis.csv ref=ppm_o2 var=raw-->
- Contra o PPM o3: 78,2 bits [70,7; 86,0] <!--gate-cost csv=single_analysis.csv ref=ppm_o3 var=raw-->
- **Leitura:** nenhum ponto medido domina o esquema na faixa de ~8 a ~25 KB; a partir do o1, todo PPM do domínio (salvo o o2 podado a 16 KiB, já dominado em desenvolvimento) gasta menos. A faixa entre 8 e 25 KB não tem um PPM de ordem 0; ele não estava no pré-registro e não foi calculado.

## Secundárias (rotuladas)

- **Reserva Unishox2 no lugar da crua:** contra o PPM o1, 4,0 bits [-1,4; 9,6]: sinal indeterminado, não equivalente <!--gate-cost csv=single_analysis.csv ref=ppm_o1 var=unishox2-->
- **A — receptor n-grama ordem 5 nos mesmos bits:** 723/1000 entregues de primeira, 140,9 bits do esquema por mensagem, 1,55 s por mensagem <!--gate csv=single_receiver_ngram.csv-->
  - O LLM no receptor acrescenta 134 entregas em 1.000 (85,7% contra 72,3%).
  - Custo para 100% com o receptor n-grama, contra o PPM o1: 70,9 bits [60,8; 81,2] <!--gate-cost csv=single_analysis.csv ref=ppm_o1 var=raw_ngram-->
- **B — busca maior no receptor antes do NACK** (feixe 256 e, se falhar, 1.024; só nas 143 falhas): 72/240 tentativas entregaram (46 de 143 no feixe 256; 26 de 97 no 1.024) <!--gate csv=single_retry.csv nobits=1-->
  - Entrega total sem NACK: 929/1.000 (Wilson 91,1%–94,3%).
  - Custo para 100% contra o PPM o1: -7,7 bits [-13,2; -1,8]: **gasta menos que o o1** <!--gate-cost csv=single_analysis.csv ref=ppm_o1 var=raw_retry-->
  - Preço: 46,1 s por tentativa no feixe 256 e 191,4 s no 1.024; em média 25,2 s a mais de receptor por mensagem (sobre todas as 1.000), concentrados nas 14,3% que falharam.
  - Contra o PPM o2 cheio, com B: +26,3 bits [20,7; 32,1]; contra o o3: +52,1 [46,5; 57,9].

## Tempo no ar por enlace (secundário; suposições de quadro de 5.28/5.41; NACK como quadro de 2 B)

Média por mensagem para entregar 100%. Segundos, salvo RockBLOCK (créditos). No APRS, o texto cru vai como caracteres.

| Enlace | Esquema + cru | Esquema + Unishox2 | Esquema + B + cru | Cru | smaz2 | Unishox2 | PPM o1 | PPM o2 | PPM o3 |
|---|---|---|---|---|---|---|---|---|---|
| Meshtastic LongFast | 0,693 | 0,679 | 0,612 | 0,719 | 0,620 | 0,622 | 0,566 | 0,534 | 0,508 |
| US915 DR0 | 0,999 | 0,941 | 0,860 | 1,590 | 1,156 | 1,161 | 0,891 | 0,751 | 0,675 |
| EU868 DR0 | 2,293 | 2,214 | 2,016 | 2,700 | 2,132 | 2,134 | 1,887 | 1,748 | 1,642 |
| APRS (2 digis, TXD 0,3 s) | 0,992 | 0,987 | 0,881 | 0,908 | 0,868 | 0,869 | 0,807 | 0,772 | 0,745 |
| RockBLOCK (créditos) | 1,310 | 1,286 | 1,154 | 1,155 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 |

- Sem B, o esquema perde para o PPM o1 nos 5 enlaces e para o Unishox2 em 4 (só ganha no US915). A vantagem em bits contra o Unishox2 some porque cada falha paga um quadro de NACK e um de reserva.
- Com B, o esquema só ganha do o1 no US915 (0,860 contra 0,891 s), pelo degrau de pacote de 11 B.

## Receptor

- Primária: 11,86 s por mensagem em média, mediana 11,50 s (Apple M4, MPS fp16); 11,54 s nas entregues e 13,78 s nas falhas.
- Memória: RSS máximo de 2,41 GB e memória do driver MPS de até 5,59 GB.
- Correção 7: a lista de candidatos foi truncada pelo teto (`cap_truncou_lista`) em 1000 de 1000 mensagens; a busca esgotou o teto de etiqueta (`busca_esgotada`) nas 143 falhas da primária e em 142 das 277 falhas de A.

## Desvios e incidentes da execução (declarados)

- **Duas interrupções da primária**, em 2026-11-05: SIGTERM em 727/1.000 (16:09) e SIGKILL em 729/1.000 (16:10). A causa foi outra sessão do Claude Code iniciando um Qwen local no mesmo Mac, a pedido do usuário em outro trabalho. Retomada com `--resume`, como o pré-registro permite: só mensagens sem linha completa foram decodificadas; nenhuma linha foi refeita nem descartada; uma revisão do modelo, uma versão de bibliotecas e um dispositivo (MPS) em todas as linhas.
- **Tempo contaminado** na faixa 650–728 pela execução concorrente: média 15,77 s, mediana 12,10 s, contra 11,52/11,46 s no resto; a mensagem 724 levou 304 s (falhou). O resultado de entrega não depende de carga: o decodificador não tem limite de tempo e o orçamento do cache KV é fixo. Só os segundos são afetados; a mediana é a leitura robusta.
- **Análise provisória** da primária e de A às 18:01 de 2026-11-05, com B em execução; a decisão de rodar B foi registrada às 16:20, antes de qualquer cálculo do endpoint.

## Arquivos

- `results/single_primary.csv`, `results/single_receiver_ngram.csv`, `results/single_retry.csv` (decodificação; versionados com `git add -f`).
- `docs/medidas/frontier_test.csv` (bits dos concorrentes), `docs/medidas/single_analysis.csv` (por mensagem, com custo e correção 7), `docs/medidas/single_summary.txt` (todas as diferenças, IC e tempo no ar).
- Gate: `python -I scripts/audit_gate.py` (marcadores `gate` e `gate-cost` acima).
