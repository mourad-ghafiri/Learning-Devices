import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from device_1 import Device_1
from device_2 import Device_2
from device_3 import Device_3

x = torch.tensor([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.float32).view(-1, 1)
y = torch.tensor([49, 36, 25, 16, 9, 4, 1, 0, 1, 4, 9, 16, 25, 36, 49], dtype=torch.float32).view(-1, 1)

class MultiDevice(torch.nn.Module):
    def __init__(self):
        super(MultiDevice, self).__init__()
        self.device_1 = Device_1()
        self.device_2 = Device_2()
        self.device_3 = Device_3()
        
        # define 2 weights
        self.w1 = torch.nn.Parameter(torch.randn(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        x1 = self.device_1(x)
        x2 = self.device_2(x)

        similarity = F.cosine_similarity(x1.view(-1), x2.view(-1), dim=0)
        # similarity = 0.2
        y = similarity * x1 + (1 - similarity) * x2

        # y = self.w1 * x1 + self.w2 * x2
        return self.device_3(y)

model = MultiDevice()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
loss_fn = torch.nn.MSELoss()

for epoch in range(10000):
    y_pred_1 = model(x)
    loss_1 = loss_fn(y_pred_1, y)
    optimizer.zero_grad()
    loss_1.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print('Epoch: {}, Loss: {}'.format(epoch, loss_1.item()))

y_hat = model(x)
device_1_y_hat = model.device_1(x)
device_2_y_hat = model.device_2(x)
device_3_y_hat = model.device_3(x)
plt.plot(x.numpy(), y.numpy(), 'b-', label='Original')
plt.plot(x.numpy(), y_hat.detach().numpy(), 'r-', label='Predicted')
plt.plot(x.numpy(), device_1_y_hat.detach().numpy(), 'o-', label='device_1_y_hat')
plt.plot(x.numpy(), device_2_y_hat.detach().numpy(), 'o-', label='device_2_y_hat')
plt.plot(x.numpy(), device_3_y_hat.detach().numpy(), 'o-', label='device_3_y_hat')
plt.legend()
plt.show()
