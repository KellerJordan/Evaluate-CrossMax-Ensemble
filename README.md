# Evaluate-Robust-Ensemble

This code evaluates the robust accuracy of the `Robust Ensemble` defense technique proposed by [Fort et al. (2024)](https://arxiv.org/pdf/2408.05446).

We evaluate the new technique using a transfer attack from a standard ensemble. This reduces the
robust accuracy of the Robust Ensemble from ~75% to ~14%.


`python evaluate_defense.py`

```
Training 10 models for use in standard and robust ensemblees...                                                                                                                                                     
100%|█████████████████████████| 10/10 [00:53<00:00,  5.32s/it]
Generating first batch of adversarial examples using PGD against the robust ensemble...
100%|███████████████████████| 100/100 [00:04<00:00, 20.22it/s]
Accuracy on first batch of adversarial examples:
Robust ensemble: tensor(0.7180)
Standard ensemble: tensor(0.7380)
Generating second batch of adversarial examples using PGD against the standard ensemble...
100%|███████████████████████| 100/100 [00:04<00:00, 21.42it/s]
Accuracy on second batch of adversarial examples:
Robust ensemble: tensor(0.1320)
Standard ensemble: tensor(0.1280)
```

![](imgs/fort2024a.png)
![](imgs/fort2024b.png)


