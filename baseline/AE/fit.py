from torch import nn
from torch.autograd import Variable

def fit(train_loader, model, optimizer, epochs, print_loss, device):
    L1_criterion = nn.L1Loss(reduction='mean')

    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            X, _ = data
            X.to(device)
            X = Variable(X)

            #clear gradient
            model.zero_grad()

            _, X_hat = model(X)

            # loss forward (reconstruction loss)
            loss = L1_criterion(X, X_hat)

            # backward and upgrade
            loss.backward()
            optimizer.step()

            if (i % 50 == 0) & print_loss:
                print('[%d/%d] [%d/%d] Loss: %.4f' % (epoch + 1, epochs, i, len(train_loader), loss))