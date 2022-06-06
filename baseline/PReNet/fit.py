import torch
from torch.autograd import Variable
from baseline.PReNet.utils import sampler_pairs

def fit(X_train_tensor, y_train, model, optimizer, epochs, batch_num, batch_size,
         s_a_a, s_a_u, s_u_u, device=None):
    # epochs
    for epoch in range(epochs):
        # generate the batch samples
        X_train_loader, y_train_loader = sampler_pairs(X_train_tensor, y_train, epoch, batch_num, batch_size,
                                                       s_a_a=s_a_a, s_a_u=s_a_u, s_u_u=s_u_u)
        for i in range(len(X_train_loader)):
            X_left, X_right = X_train_loader[i][0], X_train_loader[i][1]
            y = y_train_loader[i]

            #to device
            X_left = X_left.to(device); X_right = X_right.to(device); y = y.to(device)
            # to variable
            X_left = Variable(X_left); X_right = Variable(X_right); y = Variable(y)

            # clear gradient
            model.zero_grad()

            # loss forward
            score = model(X_left, X_right)
            loss = torch.mean(torch.abs(y - score))

            # loss backward
            loss.backward()
            # update model parameters
            optimizer.step()