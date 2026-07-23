# Analisi ciclica Caruso - versione documentale

## Installazione

```bash
pip install yfinance pandas numpy matplotlib
```

## Avvio

```bash
python caruso_analysis.py ENI.MI
python caruso_analysis.py AAPL --period 15y --plot
```

## Contenuto

Lo script implementa le formule pubblicate nei paper forniti:

- KEY
- XTL
- Composite Momentum
- livelli `0`, `+/-50`, `+/-80`
- analisi annuale, trimestrale, mensile e settimanale
- flessi/giunture settimanali
- matrice operativa multi-timeframe dei 12 casi

## Limite importante

Non è possibile dichiarare che il programma replichi al 100% l'intera tecnica
proprietaria di Francesco Caruso. Nei documenti forniti non compare la formula
completa dell'Investitore Disciplinato (ID), sebbene ne siano spiegati scopo,
stati e modalità d'impiego. Il programma replica invece la parte matematica
del Composite Momentum resa pubblica nei documenti e la relativa matrice
operativa.

Anche la semantica interna di `Stochastic[5,3]` dipende dalla piattaforma.
Lo script usa la convenzione standard raw %K(5), smussato con SMA(3), poi
applica la WMA(3) prevista dalla formula XTL.
