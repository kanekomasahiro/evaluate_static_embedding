# evaluate_static_embedding

Evaluate word embeddings using the following benchmarks.
- MEN
- SimLex
- MTurk
- MC
- WS
- RW
- RG
- SCWS
- BEHAVIOR
- Google
- MSR

## Requirements

```
pip install -u requirements.txt
```
Requires Python 3.9+

## How to evaluate

```
python evaluate.py --embedding your/embeddings/path --benchmarks men,simlex,mturk,mc,ws,rw,rg,scws,behavior,google,msr
```
