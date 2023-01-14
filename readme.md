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

mention image, 1 layer, ~10 epoch, valid:

```
top-1: 0.66433  top-5: 0.90250  top-10: 0.95014 top-20: 0.97655 top-50: 0.99038
```

whole sentence as mention representation:

```
top-1: 0.67250  top-5: 0.90795  top-10: 0.95248 top-20: 0.97754 top-50: 0.99020
```

# Explanation for 1st model

1. Why this time much higher?
   Symmetric: mention encoder and entity encoder are the same. In theroy, same inputs yield to same outputs.
   So if mention name and entity name is identical, then their similarity will be 1.
   Therefore, top-1 is always correct if mention name = entity name, which is not rare in dataset.
2. Why lower than step 1 result?
   Actually the model is not fully symmetric: we cancatenate entities into 1 sentence separated with SEP.
   This kind of output for entities diverge slightly from an independent sentence:

![](/home/xsy/snap/marktext/9/.config/marktext/images/2023-01-08-16-37-05-image.png)

# Problems & future work

1. [x] Finetune won't work: as the loss drops also do the metrics. Loss function (BCE) may be problematic.
2. [x] What will happen if encode the whole mention instead of its name only (with 1. fixed)
3. [x] Add transformer
4. [ ] Add image
