# SPOBD

Il Graph Learning è la disciplina è la disciplina che, a partire dall’osservazione di segnali registrati sui nodi di un sistema, cerca di ricostruire la struttura invisibile delle relazioni
che ha generato o regola quei segnali. Il progetto confronta un metodo iterativo classico, **PDS** (Primal-Dual Splitting), con una sua versione _Deep Unfolding_ (**Unrolling** e **Recurrent Unrolling**), su due famiglie di grafi sintetici: **Barabási-Albert (BA)** ed **Erdős-Rényi (ER)**.

## Pipeline

1. **Generazione dati** (`src/graphGenerator.py`): crea grafi sintetici connessi (BA o ER, N=50 nodi), simula segnali sul grafo e salva le coppie (distanze tra segnali, pesi degli archi) in `generated_data/`.
2. **Addestramento** (`src/train.py`): tara PDS via grid search e usa i parametri trovati per inizializzare Unrolling / Recurrent Unrolling, allenati su più profondità (`T = 3, 5, 8, 12, 20`). Checkpoint in `trained_models/`.
3. **Valutazione e grafici** (`src/curves.py`): confronta PDS, Recurrent Unrolling e Unrolling in GMSE sul test set, e produce grafici/tabelle (anche di confronto fra BA ed ER) in `figures/` e `data_for_plot/`.

`src/main.py` esegue l'intera pipeline in sequenza per entrambe le famiglie di grafo.

## Avvio

Requisiti: Python 3, con `torch`, `networkx`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`.

```bash
pip install torch networkx numpy scipy scikit-learn matplotlib
```

Tutti gli script vanno lanciati dalla **root del progetto**:

```bash
python src/main.py
```
