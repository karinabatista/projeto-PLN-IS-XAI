# Impactos da Seleção de Instâncias na Localidade e Explicabilidade

Projeto da disciplina de Processamento de Linguagem Natural (PLN), UFMG.

Este repositório contém o código e os notebooks do estudo sobre o impacto dos métodos de Seleção de Instâncias (E2SC e biO-IS) na localidade semântica e na explicabilidade de modelos BERT aplicados à classificação de textos do conjunto AGNews.

## Resumo

Investigamos se técnicas de Seleção de Instâncias (IS), originalmente voltadas à eficiência computacional, também alteram a organização semântica do espaço representacional aprendido por modelos BERT. Treinamos três modelos (baseline com treino completo, E2SC e biO-IS), extraímos os embeddings do token [CLS] e caracterizamos a localidade das vizinhanças por kNN (k=30, distância do cosseno), seguindo a metodologia de Soares (2025).

**Principal achado:** a IS tem um efeito dissociado sobre a localidade. Os dois métodos aproximam os vizinhos (maior similaridade), mas a concentração em regiões de alta homogeneidade aumenta apenas com o biO-IS (70,5% contra 65,4% do baseline), enquanto o E2SC a reduz (50,2%). A remoção de ruído do biO-IS limpa as fronteiras de classe, enquanto o E2SC, focado em redundância, compacta o espaço mas eleva a entropia média. A eficácia se mantém nos dois, com Macro-F1 igual ou superior ao baseline usando 17 a 21% menos dados de treino.

## Estrutura do repositório

```
projeto-PLN-IS-XAI/
├── README.md
├── notebooks/
│   ├── 01-treino-e-extracao.ipynb     # Treino dos modelos E2SC e biO-IS + extração dos embeddings CLS
│   ├── 02-localidade.ipynb            # Caracterização da localidade (kNN), heatmaps, tabela das 16 instâncias, métricas
│   └── 03-wordclouds.ipynb            # Nuvens de palavras das vizinhanças (baseline, E2SC, biO-IS)
└── resultados/
    ├── heatmaps/                      # Mapas de homogeneidade (entropia × distância)
    └── wordclouds/                    # Figuras comparativas das nuvens de palavras
```

## Metodologia (resumo)

- **Dataset:** AGNews (4 classes: World, Sports, Business, Sci/Tech), fold 0. Treino: 91.872 / Validação: 25.520 / Teste: 10.208.
- **Seleção de Instâncias:** E2SC (redução de redundância, β=25%) e biO-IS (redundância β=25% + ruído γ=50%), com representação TF-IDF e classificador auxiliar de Regressão Logística.
- **Modelos:** fine-tuning de `bert-base-uncased` (2 épocas, lr 2e-5, max_length 256, FP16).
- **Localidade:** kNN (k=30, cosseno), entropia de Shannon (homogeneidade) e distância média (similaridade), normalizadas min-max por cenário.
- **Explicabilidade:** nuvens de palavras com os 25 termos mais frequentes das vizinhanças.

## Dados

Os arquivos de dados (embeddings `.pkl`, modelos treinados, splits do AGNews) **não estão neste repositório** por questões de tamanho. Os notebooks foram desenvolvidos no Kaggle e referenciam datasets daquela plataforma. Para reproduzir:

1. O conjunto AGNews (fold 0) segue o split do benchmark utilizado.
2. Os embeddings são gerados pelo notebook `01-treino-e-extracao.ipynb`.
3. As métricas e figuras são geradas pelos notebooks `02` e `03`.

## Autores

- Karina Batista da Silva, UFMG
- Davila de Carvalho Meireles, UFMG
- Samai Soares Pereira, UFMG

## Referências

- Cunha, W. et al. (2023). *An effective, efficient, and scalable confidence-based instance selection framework for transformer-based text classification.* ACM SIGIR.
- Cunha, W. et al. (2025). *A noise-oriented and redundancy-aware instance selection framework.* ACM TOIS 43(2).
- Soares, L. (2025). *Localidade e explicabilidade na classificação automática de textos com modelos baseados em Transformers.* UFMG (manuscrito em preparação).
