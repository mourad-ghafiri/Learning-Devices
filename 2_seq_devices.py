import torch
import matplotlib.pyplot as plt

from device_1 import Device_1
from device_2 import Device_2

x = torch.tensor([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.float32).view(-1, 1)
y = torch.tensor([49, 36, 25, 16, 9, 4, 1, 0, 1, 4, 9, 16, 25, 36, 49], dtype=torch.float32).view(-1, 1)

class MultiDevice(torch.nn.Module):
    def __init__(self):
        super(MultiDevice, self).__init__()
        self.device_1 = Device_1()
        self.device_2 = Device_2()

    def forward(self, x):
        x = self.device_1(x)
        return self.device_2(x)

model = MultiDevice()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
loss_fn = torch.nn.MSELoss()

for epoch in range(30000):
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
plt.plot(x.numpy(), y.numpy(), 'b-', label='Original')
plt.plot(x.numpy(), y_hat.detach().numpy(), 'r-', label='Predicted')
plt.plot(x.numpy(), device_1_y_hat.detach().numpy(), 'o-', label='device_1_y_hat')
plt.plot(x.numpy(), device_2_y_hat.detach().numpy(), 'o-', label='device_2_y_hat')

plt.legend()
plt.show()
