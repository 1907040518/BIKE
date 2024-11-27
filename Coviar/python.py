import torch
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
