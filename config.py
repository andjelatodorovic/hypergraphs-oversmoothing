'''
data: coauthorship/cocitation
dataset: cora/dblp/acm for coauthorship and cora/citeseer/pubmed for cocitation
'''
data = "cocitation"
dataset = "cora"



'''
mediators: Laplacian with mediators (True) or Laplacian without mediators (False)
fast: FastHyperGCN (True) or not fast (False)
split: train-test split used for the dataset
'''
mediators = False
fast = False
split = 0



'''
gpu: gpu number to use
cuda: True or False
seed: an integer
'''
gpu = 3
cuda = True
seed = 5



'''
model related parameters
depth: number of hidden layers in the graph convolutional network (GCN)
dropout: dropout probability for GCN hidden layer
epochs: number of training epochs
'''
depth = 10
dropout = 0.5
epochs = 50
energy_weight = 0.1
n_hid = 16


'''
parameters for optimisation
rate: learning rate
decay: weight decay
'''
lr = 0.01
decay = 0.0005



import argparse, os, sys, inspect

def parse():
    """
    Adds and parses arguments / hyperparameters using argparse.
    """
    p = argparse.ArgumentParser(description="HyperGCN Argument Parser")
    
    p.add_argument('--data', type=str, default=data, help='data name (coauthorship/cocitation)')
    p.add_argument('--dataset', type=str, default=dataset, help='dataset name (e.g., cora/dblp/acm for coauthorship, cora/citeseer/pubmed for cocitation)')
    p.add_argument('--mediators', type=bool, default=False, help='True for Laplacian with mediators, False for Laplacian without mediators')
    p.add_argument('--fast', type=bool, default=False, help='faster version of HyperGCN (True)')
    p.add_argument('--split', type=int, default=1, help='train-test split used for the dataset')
    p.add_argument('--depth', type=int, default=depth, help='number of hidden layers')
    p.add_argument('--dropout', type=float, default=0.5, help='dropout probability for GCN hidden layer')
    p.add_argument('--lr', type=float, default=0.01, help='learning rate')
    p.add_argument('--decay', type=float, default=0.0005, help='weight decay')
    p.add_argument('--epochs', type=int, default=epochs, help='number of epochs to train')
    p.add_argument('--gpu', type=int, default=3, help='gpu number to use')
    p.add_argument('--n_hid', type=int, default=128, help='gpu number to use')
    p.add_argument('--cuda', type=bool, default=True, help='cuda for gpu')
    p.add_argument('--seed', type=int, default=5, help='seed for randomness')
    p.add_argument('--energy_weight', type=float, default=0.1)
    
    # Dummy argument to avoid errors in Jupyter notebooks that require `-f` argument
    p.add_argument('-f', type=str, default='', help='Dummy argument for Jupyter compatibility')

    return p.parse_args()


def current():
    """
    Returns the current directory path.
    """
    current = os.path.abspath(inspect.getfile(inspect.currentframe()))
    head, tail = os.path.split(current)
    return head

if __name__ == "__main__":
    args = parse()
    print(args)
