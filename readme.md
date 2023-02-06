# Data

## wikidiverse

### train

all data: 13205
cleaned data: 10913
image errors: 7389
entity missing: 25852
no matching: 2292
acc: 82.643

### valid

all data: 1552
cleaned data: 1302
image errors: 841
entity missing: 3145
no matching: 250
acc: 83.892

### test

all data: 1570
cleaned data: 1288
image errors: 908
entity missing: 3028
no matching: 282
acc: 82.038

## wikimel

### train

all data: 18092
cleaned data: 17568
mention image errors: 1
entity image errors: 192968
brief missing: 59132
no matching: 3397
mention not found: 524

### valid

all data: 2585
cleaned data: 2516
mention image errors: 0
entity image errors: 27366
brief missing: 7979
no matching: 510
mention not found: 69

### test

all data: 5169
cleaned data: 5022
mention image errors: 0
entity image errors: 55282
brief missing: 17297
no matching: 980
mention not found: 147

# Method & Result

## 1st step

edit distance

```python
>>> top(ans, 1)
0.7831541218637993
>>> top(ans, 5)
0.9185583432895261
>>> top(ans, 10)
0.9496216646754281
>>> top(ans, 20)
0.9695340501792115
>>> top(ans, 50)
0.9864595778574273
```

## 2nd step

lr=1e-3, margin=0.25

### Just bert, mention name independent

w/o finetune:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│       top-10_epoch        │     0.892274022102356     │
│        top-1_epoch        │    0.7214257121086121     │
│       top-20_epoch        │    0.9261250495910645     │
│       top-50_epoch        │    0.9649541974067688     │
│        top-5_epoch        │     0.853245735168457     │
└───────────────────────────┴───────────────────────────┘
```

w finetune (1 epoch, val result):

```
top-1:0.75300   top-5:0.93167   top-10:0.96333  top-20:0.97733  top-50:0.99100
```

### Just bert, mention in complete sentence

w finetune (1 epoch):

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│       top-10_epoch        │    0.9623655676841736     │
│        top-1_epoch        │    0.7676224708557129     │
│       top-20_epoch        │     0.980286717414856     │
│       top-50_epoch        │    0.9896455407142639     │
│        top-5_epoch        │    0.9305057525634766     │
└───────────────────────────┴───────────────────────────┘
```

### Bert + Linear after average, mention in complete sentence

w/o finetune (5 epoch)

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│       top-10_epoch        │     0.970927894115448     │
│        top-1_epoch        │    0.8261649012565613     │
│       top-20_epoch        │    0.9820788502693176     │
│       top-50_epoch        │    0.9900438189506531     │
│        top-5_epoch        │    0.9480286836624146     │
└───────────────────────────┴───────────────────────────┘
```

### Bert w/o finetune + Transformer

5 epoch, 8 layers, lr=2e-3, seed=0:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│       top-10_epoch        │    0.9366785883903503     │
│        top-1_epoch        │    0.5416168570518494     │
│       top-20_epoch        │    0.9679410457611084     │
│       top-50_epoch        │    0.9872560501098633     │
│        top-5_epoch        │    0.8757467269897461     │
└───────────────────────────┴───────────────────────────┘
```

20 epoch, 8 layers, seed=1:

```
top-1: 0.56477  top-5: 0.87039  top-10: 0.93338 top-20: 0.96918 top-50: 0.98844
```

10 epoches already reached top. No overfitting.

5 epoch, 2 layers, seed=0:

```
top-1: 0.61688  top-5: 0.89000  top-10: 0.93857 top-20: 0.96788 top-50: 0.98725
```

20 epoch, 2 layers, seed=1:

```
top-1: 0.67390  top-5: 0.90754  top-10: 0.94977 top-20: 0.97416 top-50: 0.98901
```

Still slowly accending (+0.1% every epoch for latest 6 epoches).

### Entity brief + Linear

```
top-1: 0.07969  top-5: 0.28066  top-10: 0.44008 top-20: 0.63598 top-50: 0.88593
```

### Multimodal cross-attention

mention image, 0.5 layer, ~10 epoch, valid:

```
top-1: 0.66433  top-5: 0.90250  top-10: 0.95014 top-20: 0.97655 top-50: 0.99038
```

whole sentence as mention representation:

```
top-1: 0.67250  top-5: 0.90795  top-10: 0.95248 top-20: 0.97754 top-50: 0.99020
```

with structure same with the paper, 1 layer, 7 epoch, valid:

```
top-1: 0.63189  top-5: 0.89008  top-10: 0.94184 top-20: 0.97187 top-50: 0.98937
```

with superparams same with paper, 1 layer, 7 epoch, valid:

```
top-1: 0.60776  top-5: 0.88499  top-10: 0.93379 top-20: 0.96354 top-50: 0.98763
```

### Cross-attention + entity attr

mention half cross & extract name, lr=1e-3, margin=0.4, 30 epoch:

```
top-1: 0.52449  top-5: 0.84687  top-10: 0.92254 top-20: 0.96018 top-50: 0.98726
```

margin=0.5, metion full cross & sentence maxpool, 30 epoch:

```
top-1: 0.50976  top-5: 0.82855  top-10: 0.90661 top-20: 0.95599 top-50: 0.98566
```

# Explanation for 1st model

1. Why this time much higher?
   Symmetric: mention encoder and entity encoder are the same. In theroy, same inputs yield to same outputs.
   So if mention name and entity name is identical, then their similarity will be 1.
   Therefore, top-1 is always correct if mention name = entity name, which is not rare in dataset.
2. Why lower than step 1 result?
   Actually the model is not fully symmetric: we cancatenate entities into 1 sentence separated with SEP.
   This kind of output for entities diverge slightly from an independent sentence:

# Problems & future work

1. [X] Finetune won't work: as the loss drops also do the metrics. Loss function (BCE) may be problematic.
2. [X] What will happen if encode the whole mention instead of its name only (with 1. fixed)
3. [X] Add transformer
4. [ ] Add image
