# Pré-registro — artigo único: até onde vale um emissor só de hash (2026-11-05)

**Status:** commitado **antes de qualquer decodificação** das mensagens abaixo e antes de calcular os bits dos concorrentes nelas. Uma execução (primária + secundárias A e B), sem re-run para melhorar. O resultado é publicado qualquer que seja, inclusive desfavorável ao esquema.

Desenho: `docs/superpowers/specs/2026-11-05-artigo-unico-design.md` (revisão 1). Plano: `docs/superpowers/plans/2026-11-05-artigo-unico.md`. Decisões: DECISOES 5.58 e 5.59.

**Commit do código congelado:** `4013e6f369dd56036ccd611c5981af77306f2dee`

Execução, análise, fronteira, instrumento de memória e decodificador estão nesse commit e não mudam até o resultado (o gate confere com `git diff`). Ambiente: Python 3.12, numpy 2.5.4 (o fluxo de `default_rng` não é garantido entre versões do numpy), Mac M4.

## Dados

- Fonte: Taskmaster-1, `TM-1-2019/self-dialogs.json`, falas do usuário (CC BY 4.0; atribuição em `data/README_TASKMASTER_HYBRID.md`).
- **SHA-256 do bruto:** `1e590ed0ccee279e40c2fb9e083d3b9417477c6bfe35ce5b2277167698dd858d`
- **Split `dev` oficial** (`TM-1-2019/train-dev-test/dev.csv`, baixado em 2026-11-05): `data/taskmaster_dev_ids.txt`, 770 conversas, SHA-256 `77cdd74321490a07c4bd72821718cc4ab3b4c29887a6e1603dc899b9217f95d4`. Conferido: é exatamente o conjunto das conversas fora do treino e do teste oficiais, sem interseção com nenhum dos dois.
- Treino dos concorrentes e do receptor n-grama: split oficial de treino (`data/taskmaster_train_ids.txt`), sem as 30 conversas de desenvolvimento; 68.272 falas de usuário.

## Pool e seleção (fixados antes de ver as mensagens)

- Conversas do split `dev`, menos as de desenvolvimento: 768 = 770 − 2. A terceira conversa de desenvolvimento fora do treino está no split de teste; o restante desse split (2.413 falas elegíveis, só 30 usadas) fica intocado como reserva.
- Regra: falas do usuário, `strip()`, 28 a 49 bytes, sem `\n`/`\r`, sem `#` inicial, duplicatas exatas removidas na ordem do arquivo de IDs; **excluídas as falas idênticas a alguma fala do treino**.
- **Elegíveis:** 2.517
- **Idênticas ao treino:** 141
- **Pool:** 2.376 (nenhuma já usada)
- Ordenação por SHA-256(`taskmaster-1-2019|single|seed=20261105|` + mensagem); ficam as N menores, na ordem do digest.
- **N:** 1.000 (42% do pool). N foi fixado por precisão (meia-largura esperada de ±7 bits), sem depender de nenhum número do pool.
- **Semente da seleção:** 20261105
- **SHA-256 da lista:** `45a2fb615bcb939b6d15eb167556d4d8256c8c566ce7203f8a9e485a1cdce85f` (`data/taskmaster_single_1000.txt`; a lista não foi lida antes deste commit).
- Script: `scripts/analysis/single_prereg.py` (`--check` reproduz a lista e os números acima).

**Declaração de exposição.** Na revisão 1 do desenho, um revisor calculou bits no pool inteiro (PPM o1 174,3; esquema 140,7; p* 0,906) antes deste pré-registro. Esses números não entram em N, X, no método do IC nem em nenhum parâmetro.

## Parâmetros congelados

| Item | Valor |
|---|---|
| Emissor | hashes `10:4`, `chunk_bytes` 2, `head_bytes` 8, etiqueta 32 bits, campo de tamanho de 8 bits |
| Receptor (primária) | `Qwen/Qwen2.5-0.5B`, revisão `060db6499f32faf8b98477b0a26969ef7d8b9987`, MPS fp16 (o runner recusa outro dispositivo; coluna `dispositivo`) |
| Busca (primária) | feixe 64, topk 32.768, teto de 1.024 testes de etiqueta |
| Secundária A (efeito do receptor) | mesmos bits; receptor n-grama de bytes ordem 5 do split de treino; feixe 64, topk 256, teto 1.024 |
| Secundária B (busca no receptor) | só nas falhas da primária; feixe 256 e, se falhar, feixe 1.024; topk 32.768; teto 1.024 por tentativa |
| Falso aceite | ≤ 3 × 1.024 × 2⁻³² ≈ 7,2 × 10⁻⁷ por mensagem (primária + B) |
| Overhead | 40 bits (8 de tamanho + 32 de etiqueta) em **todos** os métodos, inclusive no texto cru e na reserva |
| NACK | 2 B (16 bits) |
| Execuções | 1, sem re-run; interrompida, retoma só as mensagens sem linha completa, e só se revisão, versões das bibliotecas e dispositivo forem os mesmos |

## Endpoint primário e regra de leitura

- **Grandeza:** custo por mensagem para entregar 100% = bits do esquema + (0 se entregue; senão NACK de 16 bits + reserva crua de 8L + 40 bits).
- **Endpoint primário:** diferença pareada desse custo contra o **PPM o1 no domínio** (bits ideais + 40), média sobre as N mensagens.
- **Método do IC:** bootstrap pareado por percentil, 2,5 e 97,5; numpy `default_rng(semente)`, `integers(0, n, (reps, n))`; código em `scripts/analysis/single_analysis.py` (`paired_bootstrap_ci`). A aproximação normal sai ao lado como sensibilidade; a leitura usa o bootstrap.
- **Reamostragens:** 10.000
- **Semente do bootstrap:** 20261106
- **Margem X:** 8 bits (1 byte, a granularidade da carga útil nos enlaces da tabela), fixada por esse argumento antes de qualquer dado do pool.
- **Rótulos (cada um exaustivo):**
  - *sinal:* IC inteiro abaixo de 0 → o esquema gasta menos que aquele ponto; inteiro acima → gasta mais; contém 0 → indeterminado;
  - *equivalência:* IC inteiro dentro de [−8, +8] bits → equivalente na margem de 1 byte; senão, não.
- p (entrega na primeira tentativa) sai com IC de Wilson 95%, como componente, não como decisão.

## Secundário (rotulado como tal no resultado)

- **Fronteira inteira**, reportada qualquer que seja: diferença pareada do custo contra cada ponto, mesmo método e mesmos rótulos. Pontos: texto cru; smaz2; Unishox2; PPM o1; PPM o2 podado a 16, 32 e 64 KiB de tabelas; PPM o2 cheio; PPM o3. PPM com bits ideais (`ppm_bits`) e, como sensibilidade, +2 bits de terminação.
- **Regra de poda do o2** (só contagens do treino): entradas de contexto com 2 bytes, por contagem crescente, desempate lexicográfico de (contexto, símbolo); remove até o tamanho das tabelas no layout do instrumento C ficar ≤ alvo; o escape do PPM-C se recalcula do dicionário podado.
- **SHA-256 das contagens podadas (16/32/64 KiB):** `5286aac01cc58656b496296b43ba8fa6c455a786eefe09434930e4a8eb44d342`, `0fd3300b949781a2505e9f91eb22a32162c2b2cb61ef0209106b2ac6c40a8b10`, `e46f68476930c5168f6b16e10f80fbb62aab60159d6d0be37020ee764a3a965c`
- Em desenvolvimento, o o2 podado a 16 KiB ficou dominado pelo o1 (DECISOES 5.59, adendo da etapa 2). A regra não muda.
- **Memória do emissor:** já medida, não depende das mensagens (`docs/medidas/sender_memory.csv`, DECISOES 5.59, adendo da etapa 1).
- **Plataforma de memória:** host arm64 (Mach-O), Apple clang 21, -Os; não é MCU.
- Custo com reserva **Unishox2** (8 × bytes + 40) no lugar da crua.
- **Tempo no ar** do custo para 100% e de cada ponto, numa tabela só: Meshtastic LongFast, US915 DR0, EU868 DR0, APRS (Base91, 2 digipeaters, TXDELAY 0,3 s) e RockBLOCK (créditos). Suposições de quadro de 5.28 e 5.41; NACK como quadro de 2 B. No APRS, o texto cru (ponto "cru" e reserva crua) vai como caracteres, com a etiqueta de 32 bits em 5 caracteres Base91, como em `scripts/airtime.py`; as demais cargas binárias vão em Base91.
- **A:** p e custo com o receptor n-grama. **B:** custo próprio (a falha só conta depois das duas tentativas), mesmo IC, e a latência extra do receptor.
- Segundos por mensagem e memória do receptor (RSS máximo e memória do driver MPS registrados por linha).
- Correção 7 (`cap_truncou_lista`, `busca_esgotada`) aplicada na análise, na primária e em A, sem tocar nos CSVs.
- Revisão independente do código e deste texto antes do commit (agente revisor, 2026-11-05): os achados importantes foram corrigidos no commit congelado acima.

## Previsão (só do desenvolvimento, tm30; não é resultado)

- p entre 0,744 e 0,965 (Wilson de 27/30), ponto 0,90.
- Diferença de custo contra o PPM o1 entre −21,7 e +59,7 bits nesses extremos, ponto +2,3 (`scripts/analysis/design_review/forecast.py`).
- Leitura esperada no ponto: sinal indeterminado e fora da equivalência de 1 byte (+2,3 ± 7).
- Contra o o2: o esquema gasta mais. Contra o Unishox2 sozinho: gasta menos.

## Critério de parada

Exatamente uma execução (primária + A + B, definidas acima). Depois dela, nenhum experimento novo: ideias que surgirem vão para "trabalho futuro". O único cálculo permitido depois é contabilidade sobre bits e tempos já medidos, rotulada.

## Arquivos e comandos

- Primária: `caffeinate -i .venv/bin/python scripts/analysis/run_single.py --stage primary --list data/taskmaster_single_1000.txt --out results/single_primary.csv --resume`
- A: `caffeinate -i .venv/bin/python scripts/analysis/run_single.py --stage receiver_ngram --list data/taskmaster_single_1000.txt --out results/single_receiver_ngram.csv --resume`
- B: `caffeinate -i .venv/bin/python scripts/analysis/run_single.py --stage retry --list data/taskmaster_single_1000.txt --primary-csv results/single_primary.csv --out results/single_retry.csv --resume`
- Concorrentes (só depois deste commit): `python -I scripts/analysis/indomain_frontier.py --test data/taskmaster_single_1000.txt --prereg docs/PREREGISTRO_ARTIGO_UNICO.md` → `docs/medidas/frontier_test.csv`
- Análise: `python -I scripts/analysis/single_analysis.py` → `docs/medidas/single_analysis.csv` e `docs/medidas/single_summary.txt`
- Resultado: `docs/RESULTADO_ARTIGO_UNICO.md`. Gate: `python -I scripts/audit_gate.py`.
