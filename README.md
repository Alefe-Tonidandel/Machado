# machadinho

Mini-GPT (Transformer decoder-only) escrito em PyTorch e **treinado do zero** em romances de Machado de Assis.
Gera texto em nível de caractere, no estilo (e na ortografia) da época.

O objetivo não é gerar bom texto, e sim fazer o ciclo completo de um modelo de linguagem pequeno de forma
honesta: arquitetura, treino, avaliação em um livro que o modelo nunca viu, comparação com baselines simples
e publicação no Hugging Face.

```
data/raw/*.txt ── limpeza ──► treino  | validação (fim de cada livro)
                       └───► teste   (um livro inteiro, nunca visto)

treino ──► GPT (PyTorch) ──► best.pt   (melhor ponto na validação)
                                │
       bits por caractere no livro de teste, comparados com
       chute uniforme e com um modelo de n-gramas de caracteres
```

## Como o resultado é medido

- **Livro de teste separado.** Um livro inteiro fica fora do treino, do vocabulário e da escolha de
  hiperparâmetros. Ele mede se o modelo aprendeu o estilo ou só decorou os livros de treino.
- **Bits por caractere (bpc)** com janelas deslizantes: cada caractere é pontuado uma vez, e quase todos com
  contexto longo (`utils.plan_windows`, testado).
- **Baselines sem rede neural**: chute uniforme e n-gramas de caracteres com interpolação
  (`baseline.py`). Se o Transformer não ganha disso, ele não aprendeu nada além de contagens.
- **Melhor checkpoint pela validação**, porque modelos pequenos em corpus pequeno sobreajustam.

Os números medidos ficam em [RESULTS.md](RESULTS.md), gerado por `scripts/make_results_table.py`
a partir dos `runs/*/results.json`. Nada neste README é copiado à mão.

## Arquitetura

GPT no estilo GPT-2, em PyTorch puro (`src/machadinho/model.py`):

| Componente | Escolha |
|---|---|
| Atenção | multi-cabeça, causal, `scaled_dot_product_attention(is_causal=True)` |
| Normalização | pre-LayerNorm |
| MLP | expansão 4x com GELU |
| Embeddings | token e posição aprendidas; pesos compartilhados com a camada de saída |
| Inicialização | N(0, 0.02); projeções residuais escaladas por 1/sqrt(2 · camadas) |
| Otimizador | AdamW (decay só nas matrizes), aquecimento + cosseno, clip de gradiente |
| Tokenização | um id por caractere; vocabulário só do treino; `�` para símbolos desconhecidos |

## O que é meu e o que não é

A arquitetura segue o desenho consagrado do GPT-2 e foi inspirada no vídeo *Let's build GPT* e no nanoGPT,
de Andrej Karpathy, e no artigo *Attention Is All You Need* (Vaswani et al., 2017). O que este repositório
acrescenta é o corpus em português, a limpeza dos textos, a divisão com livro de teste, os baselines, a
avaliação por janelas, os experimentos e a publicação.

## Uso

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch            # sem GPU NVIDIA: pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,hub]"

pytest                                                   # testes (a parte do modelo exige PyTorch)

python scripts/download_corpus.py                        # baixa os livros do Projeto Gutenberg
python -m machadinho.baseline --test-book papeis_avulsos # referência com n-gramas
python -m machadinho.train --test-book papeis_avulsos --out runs/small
python -m machadinho.generate --ckpt runs/small/best.pt --prompt "Capitu" --temperature 0.8

python scripts/sweep.py --test-book papeis_avulsos       # grade de experimentos
python scripts/make_results_table.py                     # gera RESULTS.md
```

Para incluir mais livros, use `python scripts/download_corpus.py --add nome=ID`, onde `ID` é o número que
aparece na URL do livro em gutenberg.org.

## Experimentos

`scripts/sweep.py` roda seis configurações: três tamanhos de modelo (`tiny`, `small`, `base`) e três
ablações a partir do `small` (contexto de 64 e de 256 caracteres, e sem dropout), mudando uma coisa por vez.

## Decisões de projeto

- **Nível de caractere.** Vocabulário pequeno e nenhum tokenizador externo. O custo são sequências longas;
  um tokenizador de subpalavras seria o próximo passo.
- **Parágrafos desquebrados.** Os livros vêm quebrados em linhas de ~70 colunas. Juntar as linhas evita que
  o modelo gaste capacidade aprendendo a quebra.
- **Vocabulário só do treino.** O livro de teste não vaza nem pelo conjunto de símbolos.
- **Validação no fim de cada livro**, cortada em fronteira de parágrafo.
- **Ordem do n-grama escolhida na validação**, nunca no teste.

## Limitações

- As edições do Gutenberg usam a ortografia da época ("commigo", "theatro"); o modelo aprende essa ortografia.
- Poucos livros: o modelo sobreajusta e pode reproduzir trechos do treino (são obras em domínio público).
- Uma semente por configuração: diferenças pequenas entre execuções podem ser ruído.
- Em nível de caractere o modelo imita o estilo, mas não mantém coerência de longo prazo.
- Além do bpc, a qualidade do texto gerado não é medida automaticamente.

## Estrutura

```
src/machadinho/
  data.py        limpeza, divisão treino/validação/teste e tokenizador
  model.py       GPT em PyTorch
  train.py       treino, parada pelo melhor ponto e avaliação final
  evaluate.py    bits por caractere com janelas deslizantes
  baseline.py    n-gramas de caracteres
  generate.py    amostragem (temperatura e top-k)
  checkpoint.py  salvar e carregar
  utils.py       agendador de lr e plano de janelas (funções puras)
scripts/         download do corpus, grade de experimentos, tabela de resultados, publicação no Hugging Face
tests/           pytest
```

## Modelo publicado

<!-- TODO: link do modelo no Hugging Face depois de rodar scripts/push_to_hub.py -->

## Leitura dos resultados

Com 1.500 passos em CPU (cerca de 32 minutos para o modelo `small`), o Transformer **não superou** o baseline
de n-gramas de caracteres: 2,97 bpc contra 2,29 bpc no livro de teste (ordem 6). A perda de treino ficou próxima
da de validação e ainda caía no fim, o que indica **subtreino**, e não sobreajuste. O texto gerado imita a
estatística de caracteres da época, mas não forma frases coerentes.

Próximo passo: treinar por mais passos, com taxa de aprendizado maior e em GPU, e comparar de novo.
