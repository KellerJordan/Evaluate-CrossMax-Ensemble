from tqdm import tqdm
import torch
from torch import nn
import torchvision.transforms as T
import torch.nn.functional as F
import airbench

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))
normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
denormalize = T.Normalize(-CIFAR_MEAN / CIFAR_STD, 1 / CIFAR_STD)

def pgd(inputs, targets, model, r=6/255, step_size=1/255, steps=100, eps=1e-5):
    """
    L^\infty bounded PGD attack
    """
    delta = torch.zeros_like(inputs, requires_grad=True)
    normalized_r = 4 * r
    normalized_step_size = 4 * step_size
    
    for step in tqdm(range(steps)):
    
        delta.grad = None
        output = model(inputs + delta)
        loss = F.cross_entropy(output, targets, reduction='none').sum()
        loss.backward()

        # take an update step using the sign of the gradient
        delta.data -= normalized_step_size * delta.grad.sign()
        
        # project to the L^\infty ball of radius r
        delta.data = delta.data.clamp(-normalized_r, normalized_r)

        # project to pixel-space i.e. [0, 1]
        delta.data = normalize(denormalize(inputs + delta.data).clip(0, 1)) - inputs

    return delta.data

class Ensemble(nn.Module):
    """
    Standard ensemble mechanism
    """
    def __init__(self, nets):
        super().__init__()
        self.nets = nn.ModuleList(nets)
    def forward(self, x):
        xx = torch.stack([net(x) for net in self.nets])
        return xx.mean(0)

class RobustEnsemble(nn.Module):
    """
    Alternate ensembling mechanism proposed by Fort et al. (2024)
    https://arxiv.org/abs/2408.05446
    """
    def __init__(self, nets):
        super().__init__()
        self.nets = nn.ModuleList(nets)
    def forward(self, x):
        xx = torch.stack([net(x) for net in self.nets])
        xx = xx - xx.amax(dim=2, keepdim=True)
        xx = xx - xx.amax(dim=0, keepdim=True)
        return xx.median(dim=0).values

if __name__ == '__main__':

    test_loader = airbench.CifarLoader('cifar10', train=False)

    print('Training 10 models for use in standard and robust ensemblees...')
    models = [airbench.train94(verbose=False) for _ in tqdm(range(10))]

    standard_ensemble = Ensemble(models).eval()
    robust_ensemble = RobustEnsemble(models).eval()

    inputs, labels = next(iter(test_loader))
    new_labels = (labels + 1 + torch.randint(9, labels.shape, device=labels.device)) % 10

    print('Generating first batch of adversarial examples using PGD against the robust ensemble...')
    adv_delta = pgd(inputs, new_labels, robust_ensemble)
    adv_inputs = inputs + adv_delta
    print('Accuracy on first batch of adversarial examples:')
    with torch.no_grad():
        print('Robust ensemble:', (robust_ensemble(adv_inputs).argmax(1) == labels).float().mean().cpu())
        print('Standard ensemble:', (standard_ensemble(adv_inputs).argmax(1) == labels).float().mean().cpu())

    print('Generating second batch of adversarial examples using PGD against the standard ensemble...')
    adv_delta = pgd(inputs, new_labels, standard_ensemble)
    adv_inputs = inputs + adv_delta
    print('Accuracy on second batch of adversarial examples:')
    with torch.no_grad():
        print('Robust ensemble:', (robust_ensemble(adv_inputs).argmax(1) == labels).float().mean().cpu())
        print('Standard ensemble:', (standard_ensemble(adv_inputs).argmax(1) == labels).float().mean().cpu())

