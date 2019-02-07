
from torch.nn import Module


class ReparametrizedSample(Module):

    def __init__(self):
        super(ReparametrizedSample, self).__init__()
        self.deterministic = False

    def deterministic_forward(self, set_deterministic):
        assert isinstance(set_deterministic, bool)
        self.deterministic = set_deterministic
