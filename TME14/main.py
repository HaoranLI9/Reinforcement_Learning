from sklearn import datasets
from utils import  *
from glow import ActNorm, AffineCoupling, InvConv1dLU
import torch.distributions.normal as normal
from check import check
n_samples = 1000
#data, _ = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=0)
data, _ = datasets.make_moons(n_samples=n_samples, shuffle=True, noise=0.05, random_state=0)

plt.figure(1)
plt.scatter(data[:, 0], data[:, 1])

flow1 = ActNorm(dim=2)
flows1 = [flow1 for _ in range(5)]
container1 = FlowModules(2, *flows1)

flow2 = InvConv1dLU(2)
flows2 = [flow2 for _ in range(5)]
container2 = FlowModules(2, *flows2)

flow3 = AffineCoupling(dim=2, parity = 0)
flows3 = [flow3 for _ in range(5)]
container3 = FlowModules(2, *flows3)

prior = normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
containers = [container1, container2, container3]
Glow = FlowModel(prior, 2, *containers)

check(torch.tensor(data),Glow)

_, output, _ = Glow.f(torch.tensor(data))
#print("type", type(output))
plt.figure(2)
plt.scatter(output[:, 0].detach().numpy(), output[:, 1].detach().numpy())

_, inv_output, _ = Glow.invf(output)
plt.figure(3)
plt.scatter(inv_output[:, 0].detach().numpy(), inv_output[:, 1].detach().numpy())

plt.show()