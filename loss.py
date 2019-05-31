from torch import nn
from torch.nn import functional as F
from pdb import set_trace as pause

class LossBinary:
    """
    Loss defined as BCE - log(soft_jaccard)

    Vladimir Iglovikov, Sergey Mushinskiy, Vladimir Osin,
    Satellite Imagery Feature Detection using Deep Convolutional Neural Network: A Kaggle Competition
    arXiv:1706.06169
    """
    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss().cuda()
        #self.nll_loss = nn.CrossEntropyLoss().cuda()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        pause()
        loss = (1 - self.IoU(outputs, targets)) * self.nll_loss(outputs.argmax(dim=1), targets)

        #if self.jaccard_weight :
        #    loss += self.jaccard_weight * (1 - soft_jaccard(outputs, targets))
        
        return loss

    def IoU(self, input, target) :
        '''Intersection-over-Union (IoU)
        https://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf
        '''
        eps = 1e-15
        interception = (input.argmax(dim=1) * target).sum()
        union = (input.argmax(dim=1) + target).sum()

        return interception/(union - interception + eps)
