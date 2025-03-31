from torchvision.datasets import Imagenette

dataset = Imagenette(root='./data', split='train', download=True,size='320px')
dataset = Imagenette(root='./data', split='val', download=True,size='320px')
