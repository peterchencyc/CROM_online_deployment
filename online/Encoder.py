class Encoder(object):
    def __init__(self, network):
        self.network = network

    def forward(self, x):
        return self.network(x)