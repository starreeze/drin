# Method

symmetric: mention **name** and entity name feed into bert, then calculate cosine similarity for mean value of their token representations.

# Result

1st step result:

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

bert w/o finetune:

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

# Explanation

1. Why this time much higher?
   Symmetric: mention encoder and entity encoder are the same. In theroy, same inputs yield to same outputs.
   So if mention name and entity name is identical, then their similarity will be 1.
   Therefore, top-1 is always correct if mention name = entity name, which is not rare in dataset.
2. Why lower than step 1 result?
   Actually the model is not fully symmetric: we cancatenate entities into 1 sentence separated with SEP.
   This kind of output for entities diverge slightly from an independent sentence:

![](/home/xsy/snap/marktext/9/.config/marktext/images/2023-01-08-16-37-05-image.png)

# Problems & future work

1. Finetune won't work: as the loss drops also do the metrics. Loss function (BCE) may be problematic.

2. What will happen if encode the whole mention instead of its name only (with 1. fixed) 
