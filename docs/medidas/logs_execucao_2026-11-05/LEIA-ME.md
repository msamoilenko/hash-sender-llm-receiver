# Logs da execução pré-registrada do artigo único (2026-11-05/06)

Saída bruta do `scripts/analysis/run_single.py`, copiada do scratchpad da sessão para não se perder. Prova das interrupções e da retomada declaradas em `docs/RESULTADO_ARTIGO_UNICO.md` ("Desvios e incidentes"). Verificado antes de versionar: sem credenciais.

| Arquivo | Estágio | O que mostra |
|---|---|---|
| `run_primary.log` | primária, 1ª passada (13:43) | mensagens 0 a 726; termina sem traceback (SIGTERM às 16:09, cadeia com saída 143) |
| `run_primary2.log` | primária, 1ª retomada (16:09) | começa na 727, decodifica 727 e 728; SIGKILL às 16:10 (saída 137) |
| `run_primary3.log` | primária, 2ª retomada (16:18) | começa na 729 e vai até a 999 |
| `run_ngram.log` | secundária A | as 1.000 mensagens com o receptor n-grama |
| `run_retry.log` | secundária B | as 143 falhas da primária, feixe 256 e depois 1.024 |

Causa das interrupções: outra sessão do Claude Code iniciou um Qwen local no mesmo Mac. Os CSVs em `results/single_*.csv` são a fonte dos números; os logs são só a trilha da execução.
