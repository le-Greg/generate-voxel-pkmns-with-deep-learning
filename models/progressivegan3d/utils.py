# From https://github.com/facebookresearch/pytorch_GAN_zoo

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def loadPartOfStateDict(module, state_dict, forbiddenLayers=None):
    r"""
    Load the input state dict to the module except for the weights corresponding
    to one of the forbidden layers
    """
    own_state = module.state_dict()
    if forbiddenLayers is None:
        forbiddenLayers = []
    for name, param in state_dict.items():
        if name.split(".")[0] in forbiddenLayers:
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data

        own_state[name].copy_(param)


def loadStateDictCompatible(module, state_dict):
    r"""
    Load the input state dict to the module except for the weights corresponding
    to one of the forbidden layers
    """
    own_state = module.state_dict()
    for name, param in state_dict.items():
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data

        if name in own_state:
            own_state[name].copy_(param)
            continue

        # Else see if the input name is a prefix
        suffixes = ["bias", "weight"]
        found = False
        for suffix in suffixes:
            indexEnd = name.find(suffix)
            if indexEnd > 0:
                newKey = name[:indexEnd] + "module." + suffix
                if newKey in own_state:
                    own_state[newKey].copy_(param)
                    found = True
                    break

        if not found:
            raise AttributeError("Unknow key " + name)


def finiteCheck(parameters):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    for p in parameters:
        infGrads = isinf(p.grad.data)
        p.grad.data[infGrads] = 0

        nanGrads = isnan(p.grad.data)
        p.grad.data[nanGrads] = 0
