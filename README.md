# CoCES

CoCES (Compact Counterfactual Evidence Subgraph Learning) is designed for
knowledge graph-augmented large language model reasoning.

## Problem

Candidate subgraphs retrieved from a knowledge graph often contain redundant
paths, incorrectly directed paths, hub shortcuts, and semantically similar
paths leading to wrong answers. Even when the final answer is correct, it can
be difficult to determine which paths actually support that answer.

CoCES selects compact and sufficient evidence from a high-recall candidate path
set. It verifies path contribution through counterfactual deletion and produces
a locally non-deletable evidence set.

## Workflow

1. Perform bounded multi-hop search from the topic entities.
2. Construct candidate paths and answers using relation relevance, answer type,
   path direction, and hub penalties.
3. Use a path selector to estimate the selection probability of each path.
4. Use an answer-support evaluator to score candidate answers under the current
   evidence.
5. Train with answer ranking, sparsity, counterfactual deletion, distractor
   suppression, and weak supervision.
6. During inference, attempt to remove paths in ascending order of deletion
   contribution.
7. Return the compressed evidence while preserving the predicted answer,
   support threshold, and ranking margin.

## Running the Project

### Installation

```bash
cd git_coces
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Data Preparation

WebQSP:

```bash
coces prepare \
  --dataset webqsp \
  --input data/raw/webqsp/WebQSP.train.json \
  --triples data/kg/freebase.tsv \
  --names data/kg/entity_names.tsv \
  --types data/kg/entity_types.tsv \
  --relations data/kg/relation_names.tsv \
  --max-hops 3 \
  --max-paths 100 \
  --output data/processed/webqsp/train.jsonl
```

CWQ:

```bash
coces prepare \
  --dataset cwq \
  --input data/raw/cwq/ComplexWebQuestions_train.json \
  --triples data/kg/freebase.tsv \
  --names data/kg/entity_names.tsv \
  --types data/kg/entity_types.tsv \
  --relations data/kg/relation_names.tsv \
  --max-hops 4 \
  --max-paths 100 \
  --output data/processed/cwq/train.jsonl
```

Run the same commands for the development and test splits with the
corresponding input and output paths.

### Training

```bash
coces train --config configs/webqsp.yaml
```

or:

```bash
coces train --config configs/cwq.yaml
```

### Inference

```bash
coces predict \
  --checkpoint outputs/webqsp/final \
  --input data/processed/webqsp/test.jsonl \
  --output outputs/webqsp/predictions.jsonl
```

Enable natural-language answer generation:

```bash
coces predict \
  --checkpoint outputs/webqsp/final \
  --input data/processed/webqsp/test.jsonl \
  --output outputs/webqsp/predictions.jsonl \
  --generate
```

### Evaluation

```bash
coces evaluate \
  --predictions outputs/webqsp/predictions.jsonl \
  --output outputs/webqsp/metrics.json
```

The evaluation reports Hits@1, F1, Average Evidence Size (AES), and Local
Irreducibility Rate (LIR).

### Ablation Experiments

```bash
python scripts/make_ablations.py \
  --base configs/webqsp.yaml \
  --output-dir configs/ablations/webqsp

python scripts/run_ablations.py \
  --config-dir configs/ablations/webqsp \
  --test-file data/processed/webqsp/test.jsonl \
  --output-dir outputs/ablations/webqsp
```
