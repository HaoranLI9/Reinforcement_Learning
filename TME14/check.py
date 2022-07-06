from tkinter import X
from matplotlib import container
from utils import *
import torch.distributions.normal as normal
def check(x: torch.Tensor, mm):
  #print(x)
  #p_x = prior.log_prob(x)
  _, y, det_y = mm.f(x)
  #print(type(y))
  _, x1, det_x = mm.invf(y)
  #print(det_x, det_y)
  #error_p = abs(p_x1-p_x).sum()
  error = abs(x1-x).sum()
  #print(error, error_p)
  print(error, abs(det_y + det_x))
  if error < 1e-3 and abs(det_y + det_x) < 1e-3:
    print(True)
  else:
    print(False)

x = torch.rand(100,10) #100 batch / 10 dim
flow = AffineFlowModule(10)
flows = [flow for _ in range(1)]
container = FlowModules(10, *flows)


prior = normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
containers = [container for _ in range(2)] 
model = FlowModel(prior, 10, *containers)
if __name__ == '__main__':
  check(x, model)
  
