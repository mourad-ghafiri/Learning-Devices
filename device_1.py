import torch
import matplotlib.pyplot as plt

x = torch.tensor([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.float32).view(-1, 1)
y = torch.tensor([49, 36, 25, 16, 9, 4, 1, 0, 1, 4, 9, 16, 25, 36, 49], dtype=torch.float32).view(-1, 1)

class Device_1(torch.nn.Module):
    def __init__(self):
        super(Device_1, self).__init__()
        self.linear_1 = torch.nn.Linear(1, 10)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = self.relu(self.linear_1(x))
        return self.linear_2(x)

model = Device_1()
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
