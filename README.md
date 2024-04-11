For this experiment, I have created four models. Each
model was a CNN trained on the dataset CIFAR-10. Two of
these models are inspired by a VGGnet architecture and two
of them follow a Resnet architecture; the convolutional part
is the same for each model, only changing with those that
include skip connections. The models have the same number
of convolutional layers (with the same pooling layer) and the
same number of filters in each layer; the only difference is
that in two of the models, the layers include skip connections.
After the convolutional part, two models (one VGG and the
other Resnet) have six dense layers, while the other two have
twelve dense layers; all the layers have the same number of
neurons.

Before running the Main code, remember to install the dependencies:

```
pip install -r requirements.txt
```

To run the Main code, use:

```

python main.py run
```
I would advise to create a new virtual environment:

```
python -m venv /path/to/new/virtual/environment
```

