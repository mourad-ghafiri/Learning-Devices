import torch
import matplotlib.pyplot as plt

x = torch.tensor([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.float32).view(-1, 1)
y = torch.tensor([49, 36, 25, 16, 9, 4, 1, 0, 1, 4, 9, 16, 25, 36, 49], dtype=torch.float32).view(-1, 1)

class Device_3(torch.nn.Module):
    def __init__(self):
        super(Device_3, self).__init__()
        self.linear_1 = torch.nn.Linear(1, 5)
        self.sig = torch.nn.Sigmoid()
        self.linear_2 = torch.nn.Linear(5, 5)
        self.gelu = torch.nn.GELU()
        self.linear_3 = torch.nn.Linear(5, 1)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.sig(x)
        x = self.linear_2(x)
        x = self.gelu(x)
        x = self.linear_3(x)
        return x

model = Device_3()
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
plt.plot(x.numpy(), y.numpy(), 'b-', label='Original')
plt.plot(x.numpy(), y_hat.detach().numpy(), 'r-', label='Predicted')

plt.legend()
plt.show()
