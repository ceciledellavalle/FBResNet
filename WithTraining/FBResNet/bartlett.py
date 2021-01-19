import torch

# torch type and device
def Test_cuda(verbose = False):
    use_cuda = torch.cuda.is_available()
    if verbose:    
        print("Use cuda : ",use_cuda)
    if use_cuda:
        torchdevice = torch.device('cuda:1')
        torch.cuda.set_device(torchdevice)
        torchdtype = torch.float32
    else :
        torchdevice = 'cpu'
        torchdtype = torch.float32
    return torchdevice, torchdtype