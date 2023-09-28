import torch

t = torch.Tensor([3.9754e-01, 2.5118e-01, 1.1507e-01, 8.5251e-02, 3.3587e-02, 2.2410e-02,
        1.2181e-02, 9.3503e-03, 8.6414e-03, 7.2640e-03, 7.1975e-03, 6.8097e-03,
        5.1380e-03, 4.2904e-03, 3.8153e-03, 3.5570e-03, 3.0007e-03, 2.5742e-03,
        2.3237e-03, 2.1737e-03, 1.4150e-03, 1.2533e-03, 1.2384e-03, 1.1958e-03,
        1.1351e-03, 1.1113e-03, 1.0378e-03, 7.3687e-04, 7.2170e-04, 6.9898e-04,
        5.8916e-04, 5.6449e-04, 5.5803e-04, 5.2136e-04, 4.7917e-04, 4.7438e-04,
        4.7177e-04, 4.4015e-04, 4.1495e-04, 4.0984e-04, 3.9758e-04, 3.9028e-04,
        3.8579e-04])

for i in range(100):
    item = torch.multinomial(t, num_samples=1, replacement=True)
    print(item)