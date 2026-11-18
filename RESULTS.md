# Resultados

Livro de teste (nunca visto no treino nem na escolha de hiperparâmetros): `papeis_avulsos`.
Métrica: bits por caractere (bpc); menor é melhor. Uma semente por configuração.

| Execução | Camadas | d_model | Contexto | Dropout | Parâmetros | Val bpc | Teste bpc | Tempo (s) |
|---|---|---|---|---|---|---|---|---|
| small | 4 | 128 | 128 | 0.1 | 823,040 | 2.871 | 2.966 | 1931 |
| tiny | 2 | 64 | 64 | 0.1 | 110,848 | 3.337 | 3.368 | 232 |

## Referências sem rede neural

| Modelo | Val bpc | Teste bpc |
|---|---|---|
| Uniforme (chute aleatório entre os símbolos) | 6.700 | 6.700 |
| N-grama de caracteres (ordem 6, escolhida na validação) | 2.092 | 2.293 |

Leitura: a diferença entre `Val bpc` e `Teste bpc` mostra o quanto o modelo depende do estilo dos livros de treino. Diferenças pequenas entre execuções podem ser ruído de uma única semente.
