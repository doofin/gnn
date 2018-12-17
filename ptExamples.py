import torch

import torch.functional as F

from fn import _
# x = torch.rand(5, 3)
# print(x)
#
# inputd = torch.randn(128, 20) # random tensor of 128 * 20
#
#
# fc = torch.nn.Linear(20, 30)
# output = fc(inputd)
# print(output.size())

def myl():
    def learnF(x):return 3*x+1.0
    for i in list(range(1000)):
        x = i
        w = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(1.0, requires_grad=True)
        y=w*x + b
        mse_loss = torch.nn.MSELoss()
        loss = abs((y-torch.tensor(learnF(x)))) #mse_loss(y,torch.tensor(learnF(x)))
        opti=torch.optim.Adam([w,b])

        opti.zero_grad()
        loss.backward()
        opti.step()
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        print("real y:",learnF(x))
        print(i,"th,,pred:",y.item(),",,",loss.item())


def ln():
    from itertools import count

    import torch
    import torch.autograd
    import torch.nn.functional as F

    POLY_DEGREE = 4
    W_target = torch.randn(POLY_DEGREE, 1) * 5
    b_target = torch.randn(1) * 5

    def make_features(x):
        """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
        x = x.unsqueeze(1)
        return torch.cat([x ** i for i in range(1, POLY_DEGREE + 1)], 1)

    def f(x):
        """Approximated function."""
        return x.mm(W_target) + b_target.item()

    def poly_desc(W, b):
        """Creates a string description of a polynomial."""
        result = 'y = '
        for i, w in enumerate(W):
            result += '{:+.2f} x^{} '.format(w, len(W) - i)
        result += '{:+.2f}'.format(b[0])
        return result

    def get_batch(batch_size=32):
        """Builds a batch i.e. (x, f(x)) pair."""
        random = torch.randn(batch_size)
        x = make_features(random)
        y = f(x)
        return x, y

    # Define model
    fc = torch.nn.Linear(W_target.size(0), 1)

    for batch_idx in count(1):
        # Get data
        batch_x, batch_y = get_batch()

        # Reset gradients
        fc.zero_grad()

        # Forward pass
        output = F.smooth_l1_loss(fc(batch_x), batch_y)
        loss = output.item()

        # Backward pass
        output.backward()

        # Apply gradients
        for param in fc.parameters():
            param.data.add_(-0.1 * param.grad.data)

        # Stop criterion
        if loss < 1e-3:
            break

    print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))
    print('==> Learned function:\t' + poly_desc(fc.weight.view(-1), fc.bias))
    print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))


# ln();
myl()