import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint


class ODEBlock(nn.Module):
    """Solves ODE defined by odefunc.
    Parameters
    ----------
    odefunc : ODEFunc instance or anode.conv_models.ConvODEFunc instance
        Function defining dynamics of system.
    is_conv : bool
        If True, treats odefunc as a convolutional model.
    tol : float
        Error tolerance.
    adjoint : bool
        If True calculates gradient with adjoint method, otherwise
        backpropagates directly through operations of ODE solver.
    """
    def __init__(self, odefunc, is_conv=False, tol=1e-3, adjoint=False, max_num_steps=100000, method='dopri5'):
        super(ODEBlock, self).__init__()
        self.adjoint = adjoint
        self.is_conv = is_conv
        self.odefunc = odefunc
        self.method = method
        self.tol = tol
        self.max_num_steps = max_num_steps

    def forward(self, x, eval_times=None):
        """Solves ODE starting from x.
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, self.odefunc.data_dim)
        eval_times : None or torch.Tensor
            If None, returns solution of ODE at final time t=1. If torch.Tensor
            then returns full ODE trajectory evaluated at points in eval_times.
        """
        # Forward pass corresponds to solving ODE, so reset number of function
        # evaluations counter

        if eval_times is None:
            integration_time = torch.tensor([0, 1]).float().type_as(x)
        else:
            integration_time = eval_times.type_as(x)


        if self.odefunc.augment_dim > 0:
            if self.is_conv:
                # Add augmentation
                batch_size, channels, height, width = x.shape
                aug = torch.zeros(batch_size, self.odefunc.augment_dim,
                                  height, width, device=x.device)
                # Shape (batch_size, channels + augment_dim, height, width)
                x_aug = torch.cat([x, aug], 1)
            else:
                # Add augmentation
                aug = torch.zeros(x.shape[0], self.odefunc.augment_dim, device=x.device)
                # Shape (batch_size, data_dim + augment_dim)
                x_aug = torch.cat([x, aug], 1)
        else:
            x_aug = x

        if self.adjoint:
            out = odeint_adjoint(self.odefunc, x_aug, integration_time,
                                 rtol=self.tol, atol=self.tol, method=self.method,
                                 options={'max_num_steps': self.max_num_steps})
        else:
            out = odeint(self.odefunc, x_aug, integration_time,
                         rtol=self.tol, atol=self.tol, method=self.method,
                         options={'max_num_steps': self.max_num_steps})

        if eval_times is None:
            return out[1]  # Return only final time
        else:
            return out

    def trajectory(self, x, timesteps=None, integration_time=None):
        """Returns ODE trajectory.
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, self.odefunc.data_dim)
        timesteps : int
            Number of timesteps in trajectory.
        """
        assert timesteps is not None or integration_time is not None
        if integration_time is None:
            integration_time = torch.linspace(0., 1., timesteps)
        return self.forward(x, eval_times=integration_time)

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Conv2dTime(nn.Conv2d):
    """
    Implements time dependent 2d convolutions, by appending the time variable as
    an extra channel.
    """
    def __init__(self, in_channels, *args, **kwargs):
        super(Conv2dTime, self).__init__(in_channels + 1, *args, **kwargs)

    def forward(self, t, x):
        # Shape (batch_size, 1, height, width)
        t_img = torch.ones_like(x[:, :1, :, :]) * t
        # Shape (batch_size, channels + 1, height, width)
        t_and_x = torch.cat([t_img, x], 1)
        return super(Conv2dTime, self).forward(t_and_x)


class ConvODEFunc(nn.Module):
    """Convolutional block modeling the derivative of ODE system.

    Parameters
    ----------
    img_size : tuple of ints
        Tuple of (channels, height, width).

    nf : int
        Number of convolutional filters.

    augment_dim: int
        Number of augmentation channels to add. If 0 does not augment ODE.

    time_dependent : bool
        If True adds time as input, making ODE time dependent.

    non_linearity : string
        One of 'relu' and 'softplus'
    """
    def __init__(self, nf, nb, augment_dim=0,
                 time_dependent=False, non_linearity='relu'):
        super(ConvODEFunc, self).__init__()
        self.augment_dim = augment_dim
        self.time_dependent = time_dependent
        self.nfe = 0  # Number of function evaluations
        self.num_filters = nf + augment_dim

        if time_dependent:
            self.convs = nn.ModuleList([Conv2dTime(self.num_filters, self.num_filters,  kernel_size=3, stride=1, padding=1) for _ in range(nb)])
        else:
            self.convs = nn.ModuleList([nn.Conv2d(self.num_filters, self.num_filters, kernel_size=3, stride=1, padding=1) for _ in range(nb)])

        if non_linearity == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif non_linearity == 'softplus':
            self.activation = nn.Softplus()

    def forward(self, t, x):
        """
        Parameters
        ----------
        t : torch.Tensor
            Current time.

        x : torch.Tensor
            Shape (batch_size, input_dim)
        """
        self.nfe += 1
        out = x
        if self.time_dependent:
            for layer in self.convs:
                out = layer(t, out)
                out = self.activation(out)
        else:
            for layer in self.convs:
                out = layer(out)
                out = self.activation(out)
        return out
