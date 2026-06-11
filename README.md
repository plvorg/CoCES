# CoCES

CoCES（Compact Counterfactual Evidence Subgraph Learning）用于知识图谱增强的大语言模型推理。

## 解决的问题

知识图谱检索得到的候选子图通常包含大量冗余路径、错误方向路径、hub shortcut 和语义相似但指向错误答案的路径。即使模型最终回答正确，也难以判断哪些路径真正支持答案。

CoCES 从高召回候选路径中选择紧凑且充分的证据，并通过逐条删除路径验证其对答案的贡献，最终得到局部不可删除的证据集合。

## 方法流程

1. 从问题的主题实体出发，在知识图谱中进行有界多跳搜索。
2. 根据关系相关性、答案类型、路径方向和 hub 惩罚构造候选路径与候选答案。
3. 使用路径选择器为每条候选路径计算选择概率。
4. 使用答案支持评估器计算候选答案在当前证据下的支持分数。
5. 结合答案排序、稀疏约束、反事实删除、干扰路径抑制和弱监督训练模型。
6. 推理时按照删除贡献从小到大尝试删除路径。
7. 在答案、支持分数和排序间隔不变的前提下，输出压缩后的证据路径。

## 运行

### 安装

```bash
cd git_coces
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 数据预处理

WebQSP：

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

CWQ：

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

开发集和测试集使用相同命令分别处理。

### 训练

```bash
coces train --config configs/webqsp.yaml
```

或：

```bash
coces train --config configs/cwq.yaml
```

### 推理

```bash
coces predict \
  --checkpoint outputs/webqsp/final \
  --input data/processed/webqsp/test.jsonl \
  --output outputs/webqsp/predictions.jsonl
```

启用大语言模型生成自然语言答案：

```bash
coces predict \
  --checkpoint outputs/webqsp/final \
  --input data/processed/webqsp/test.jsonl \
  --output outputs/webqsp/predictions.jsonl \
  --generate
```

### 评测

```bash
coces evaluate \
  --predictions outputs/webqsp/predictions.jsonl \
  --output outputs/webqsp/metrics.json
```

评测指标包括 Hits@1、F1、平均证据规模（AES）和局部不可约率（LIR）。

### 消融实验

```bash
python scripts/make_ablations.py \
  --base configs/webqsp.yaml \
  --output-dir configs/ablations/webqsp

python scripts/run_ablations.py \
  --config-dir configs/ablations/webqsp \
  --test-file data/processed/webqsp/test.jsonl \
  --output-dir outputs/ablations/webqsp
```
