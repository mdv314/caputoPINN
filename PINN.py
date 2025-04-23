import sys
sys.path.insert(0, '../Utilities/')
from collections import OrderedDict
from pyDOE import lhs
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
# from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time
np.random.seed(1234)
import torch.nn as nn
from torch.autograd import Variable
import torch
import torchtyping

from typing import List

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# neural net

class NeuralNet(nn.Module):
    """Class for the Neural Network portion of PINN

    Attributes:
        layers : Pytorch layers
            The layers of the neural network
        depth : int
            The depth of the neural network
        
    
    Methods:
        forward:
            computes forward pass of neural network given an input
    """
    def __init__(self, layers: List[int]):
        """ Initializes the layers of the neural network
        Inputs:
            layers: List[int]

        Outputs:
            None
        """
        super(NeuralNet).__init__()

        # parameters
        self.depth: int = len(layers) - 1
        self.activation = nn.Tanh

        list_layers = []
        for i in range(self.depth-1):
            list_layers.append(
                ('layer_%d' % i, nn.Linear(layers[i], layers[i+1]))
            )
            list_layers.append(('activation_%d' % i, self.activation()))
        
        list_layers.append(
                ('layer_%d' % i, nn.Linear(layers[-2], layers[-1]))
            )

        layer_dict = OrderedDict(list_layers)

        self.layers = nn.Sequential(layer_dict)
    
    def forward(self, x):
        """ Implements the forward pass portion of the Neural Network
        Inputs:
            x : Any
                Input to the neural network
        Outputs:
            out : Any (change later)
                Output of the neural network
        """
        out = self.layers(x)
        return out

class PINN():
    """ Class for implementation of Physics Informed Neural Network
    Attributes:
        lb: Left boundary 
        ub: Upper boundary
        x_u: x predicted by the NN
        t_u: t predicted by the NN
        x_f: x from the function
        t_f: t from the function
        u: Neural Net prediction of u
        layers: Layers of the nn
        nu: Nu
        dnn: Neural network
        optimizer: optimizer for the training 
        iter: iteration counter
    Methods:
    """
    def __init__(self, X_u, u , X_f, layers, lb, ub, nu):
        """ Initialization of PINN
        Output: None
        """

        # boundary conditions
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        # data
        self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).float().to(device)
        self.t_u = torch.tensor(X_u[:, 1:2], requires_grad=True).float().to(device)
        self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)
        
        self.layers = layers
        self.nu = nu
        
        # deep neural networks
        self.dnn = NeuralNet(layers).to(device)
        
        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(), 
            lr=1.0, 
            max_iter=50000, 
            max_eval=50000, 
            history_size=50,
            tolerance_grad=1e-5, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"       # can be "strong_wolfe"
        )

        self.iter = 0
    
    def net_u(self, x ,t):
        """ Computes the u from the network 
        Input: 
            x: Parameter x
            t: Parameter t
        Output:
            u: prediction of the neural network
        """
        u = self.dnn(torch.cat([x,t], dim=1))
        return u

    def net_f(self, x, t):
        """Using pytorch autograd to compute residual with respect to the function
        Input:
            x: Input parameter x
            t: Input parameter t
        Output:
            f: Residual with respect to the function
        """
        u = self.net_u(x, t)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_x = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, x, 
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]
        
        f = u_t + u * u_x - self.nu * u_xx
        return f
    
    def loss_func(self):
        """ Combined loss of the equation and the prediction, meant to be given as input to an optimizer
        Input:
            None
        Output:
            loss: combined loss
        """
        self.optimizer.zero_grad()

        u_pred = self.net_u(self.x_u, self.t_u)
        f_pred = self.net_f(self.x_f, self.t_f)
        loss_u = torch.mean((self.u - u_pred) ** 2)
        loss_f = torch.mean(f_pred ** 2)
        
        loss = loss_u + loss_f
        
        loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' % (self.iter, loss.item(), loss_u.item(), loss_f.item())
            )
        return loss

    def train(self):
        """ Train the PINN
        Input:
            None
        Output:
            None
        """
        self.dnn.train()

        self.optimizer.step(self.loss_func)

    def predict(self, X):
        """ Function for testing the neural network
        Input:
            X: input vector containing both x and t
        Output:
            u: Prediction
            f: Prediction
        """
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        u = self.net_u(x, t)
        f = self.net_f(x, t)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f


if __name__ == "__main__":
    nu = 0.01/np.pi
    noise = 0.0        

    N_u = 100
    N_f = 10000
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    # x,t and usol are the main things we need
    data = scipy.io.loadmat("data/burgers_shock.mat")

    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T

    X, T = np.meshgrid(x,t)

    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]              

    # Domain bounds
    lb = X_star.min(0)
    ub = X_star.max(0)    
    
    # data preprocessing
    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
    uu1 = Exact[0:1,:].T
    xx2 = np.hstack((X[:,0:1], T[:,0:1]))
    uu2 = Exact[:,0:1]
    xx3 = np.hstack((X[:,-1:], T[:,-1:]))
    uu3 = Exact[:,-1:]

    X_u_train = np.vstack([xx1, xx2, xx3])
    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.vstack([uu1, uu2, uu3])

    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx,:]

    # model training
    model = PINN(X_u_train, u_train, X_f_train, layers, lb, ub, nu)

    model.train()
    
    # model testing
    u_pred, f_pred = model.predict(X_star)

    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    print('Error u: %e' % (error_u))                     

    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    Error = np.abs(Exact - U_pred)