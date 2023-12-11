# This file is part of RegBN: Batch Normalization of Multimodal Data 
# with Regularization.
#
# RegBN is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RegBN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RegBN. If not, see <https://www.gnu.org/licenses/>.


from typing import Union, List, Sequence
import os
import math
import operator

import torch
import torch.nn as nn
from   torch import Tensor
from functools import reduce


torchV = int(torch.__version__.split('.')[1])
if torchV < 13:
    from torch import inverse as inv_torch
    from torch import svd     as svd_torch
else:
    from torch.linalg import inv as inv_torch
    from torch.linalg import svd as svd_torch

L1torch = nn.L1Loss()
L2torch = nn.MSELoss()
epsilon_ = 1e-8


class RegBN(nn.Module):
    r""" Implements Batch Normalization via Thikhonov Regulizer (RegBN) module

    RegBN can be applied to layers in NNs as a normalization technique to remove 
    metadata  effects from the features in a network  at the batch level.

    Args:
        f_num_channels (int) – the number of channels in f data
        g_num_channels (int) – the number of channels in g data

        f_layer_dim (List[Sequence[int]]) – the spatial dimenstion of f
        g_layer_dim (List[Sequence[int]]) – the spatial dimenstion of g

        device (int) – the id of CUDA. Right now, only single gpu is supported

        beta1 (float) – beta1 in eq (6)
        beta2 (float) – beta2 in eq (6)
        momentum (float) – momentum in eq (6)

        normalize_input (bool) – normalisation of input f 
        normalize_output (bool) –  normalisation of output f 
        affine (boo): batch normalisation`s affine

        sigma_THR (float) – the threshold value for stds extracted by SVD (default: 0.)
        sigma_MIN (float) – the minimum cutoff value of std (default: 0.0)

        verbose (bool) – print some results

    Example:
        >>> batchSize = 100
        >>> f = torch.rand([batchSize, 128]).to("gpu:0")
        >>> g = torch.rand([batchSize, 16]).to("gpu:0")
        >>> kwargs = {
            'gpu': 0,
            'f_num_channels': 128, 
            'g_num_channels': 16,
            'f_layer_dim': [],
            'g_layer_dim':[],
            'normalize_input': True,
            'normalize_output': True,
            'affine': False,
        }
        >>> regbn_module = RegBN(**kwargs)
        ...

        # training:
        >>> kwargs_train = {"is_training": True, 'n_epoch': 1}
        >>> f_n, g_n = regbn_module(f, g, **kwargs_train) 

        # valid/inference:
        >>> kwargs_test = {"is_training": False}
        >>> f_n, g_n = regbn_module(f, g, **kwargs_test) 

    """
    __constants__ = ['f_num_channels', 'g_num_channels']
    f_num_channels: int
    g_num_channels: int

    def __init__(self, 
                f_num_channels: int = None,
                g_num_channels: int = None, 
                f_layer_dim: List[Sequence[int]] = None,
                g_layer_dim: List[Sequence[int]] = None,
                gpu: int = 0,
                beta1: float = 0.9, 
                beta2: float = 0.99,
                momentum: float = 0.02,
                normalize_input: bool = False, 
                normalize_output: bool = True,
                affine: bool = False,
                sigma_THR: float = 0.0, 
                sigma_MIN: float = 0.0, 
                verbose: bool = False,
        ): 
        super(RegBN, self).__init__()

        self.f_num_channels = f_num_channels
        self.g_num_channels = g_num_channels

        assert beta1 < 1, \
            ValueError("Invalid beta1: {}".format(beta1))

        assert beta2 < 1, \
            ValueError("Invalid beta2: {}".format(beta2))

        self.beta1 = beta1
        self.beta2 = beta2
        self.sigma_THR = sigma_THR
        self.sigma_MIN = sigma_MIN
        self.verbose = verbose
        self.device = f"cuda:{gpu}"

        g_dim_flat = reduce(operator.mul, [g_num_channels]+g_layer_dim, 1)
        f_dim_flat = reduce(operator.mul, [f_num_channels]+f_layer_dim, 1)

        # store lambda+ values
        self.lambda_set = torch.tensor(())
        self.is_nan_ = False

        # LBFG-solver params
        lbfgs_max_iter = 25
        lbfgs_kwargs = {'max_iter': lbfgs_max_iter,
                        'max_eval': lbfgs_max_iter* 7 // 4,
                        'history_size': lbfgs_max_iter * 20,
                        'line_search_fn': "strong_wolfe",
                        'tolerance_grad': 1e-05,
                        'tolerance_change': 1e-09,
        }
        
        self.W_calc = proj_matrix_estimator(lbfgs_kwargs)
        self.register_buffer('W', torch.zeros(g_dim_flat, f_dim_flat).to(self.device))

        # updateing projection weights
        self._reset_adam_prams(momentum)


        # normlaisation of inputs/outputs
        self.norm_f = _get_norm_inp(normalize_input, f_num_channels, f_layer_dim, affine)
        self.norm_g = _get_norm_inp(normalize_input, g_num_channels, g_layer_dim, affine)

        self.norm_f_out = _get_norm_out(normalize_output, f_num_channels, f_layer_dim, affine)
        self.norm_g_out = _get_norm_out(normalize_output, g_num_channels, g_layer_dim, affine)

    def _reset_adam_prams(self, momentum: float = 0.1, ) -> None:
        """ Prepares the seeting for updated projection weights """
        self.m = 0.
        self.v = 0.
        self.Momentum = momentum


    @torch.enable_grad()
    def update_W(self, 
                W_cur: torch.Tensor, 
                n_epoch: int, 
        ) -> None:

        with torch.no_grad():
            g = L1torch(self.W, W_cur)

            self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
            self.v = self.beta2 * self.v + (1.0 - self.beta2) * g**2

            mhat = self.m / (1.0 - self.beta1**(n_epoch+1))
            vhat = self.v / (1.0 - self.beta2**(n_epoch+1))

            eta = self.Momentum * mhat / (torch.sqrt(vhat) + epsilon_)
            self.W = (1. - eta) * self.W + eta * W_cur 


    def update_Lambda(self, lambda_, n_keep: int = 21) -> None:
        self.lambda_set = self.lambda_set[-max(n_keep, 1):]
        self.lambda_set = torch.cat((self.lambda_set, lambda_.float()), dim = 0)


    def forward(self, 
                f: torch.Tensor,
                g: torch.Tensor,
                is_training: bool = None,
                n_epoch: int = None,
                steps_per_epoch: int = None,
        ) -> torch.Tensor:
        """
        Args:
            f (Tensor) – a n-dimensional input learnable feature torch.Tensor
            g (Tensor) – a m-dimensional metadata torch.Tensor
            is_training (bool) – training or val/test
            n_epoch (int) – the current epoch
            steps_per_epoch (int) – number of steps per epoch
        """
        f_sz, g_sz = f.size(), g.size()

        if torch.isnan(f).sum() > 0 or torch.isnan(g).sum() > 0:
            return f, g


        # normlise and flatten the inputs 
        f_flat_nor = self.norm_f(f).view(f_sz[0], -1)
        g_flat_nor = self.norm_g(g).view(g_sz[0], -1) 

        if is_training:

            # svd decomposition
            g_u, g_s_diag, g_v = _svd_decomposition(g_flat_nor, 
                                                    self.sigma_THR, 
                                                    self.sigma_MIN)
            if g_u is None:
                self.is_nan_ = True
                return f, g

            # estimate W
            W_plus, lambda_, not_found = self.W_calc.compute(g_flat_nor,
                                                            g_u, 
                                                            g_s_diag, 
                                                            g_v, 
                                                            self.lambda_set)

            if (~not_found) and (len(W_plus) > 0):
                W_hat = torch.mm(W_plus, f_flat_nor)

                # update W with regards to previous batches
                self.update_W(W_hat, n_epoch)

                # update the lambda set
                self.update_Lambda(lambda_.float(), n_keep = steps_per_epoch)

            elif not_found: 
                W_hat = self.W
                if self.verbose:
                    print('not found!')


        if self.is_nan_:
            return f, g

        if not is_training:
            W_hat = self.W

        # f_mapped: feaures of `f` that mapped to the g plane
        f_mapped2g = torch.mm(g.reshape(g_sz[0], -1), W_hat) 
        f_r = f.reshape(f_sz[0], -1) - f_mapped2g
        f_r = f_r.reshape(f_sz)


        # normalise output
        f_r = self.norm_f_out(f_r)
        g   = self.norm_g_out(g)
        return f_r, g


    def extra_repr(self) -> str:
        return 'f_num_channels={f_num_channels}, ' \
                'g_num_channels={g_num_channels}'.format(**self.__dict__)




class proj_matrix_estimator(object):
    """ Computes the projection matrix in eq. (4) """
    
    def __init__(self, 
                figure: bool = False,
                pred_tolerance: float = 0.05, 
                **lbfgs_kwargs) -> None:

        self.figure = figure
        self.pred_tolerance = pred_tolerance
        self.lbfgs_kwargs = lbfgs_kwargs


    def get_usx(self, 
                x: Tensor, 
                lambda_: Tensor, 
                u: Tensor, 
                s_diag: Tensor) -> Tensor:
        sl = torch.pow(s_diag, 2) + lambda_.to(u.get_device())
        sl = torch.nan_to_num(sl, nan=epsilon_, posinf=epsilon_, neginf=epsilon_)
        sl_inv = inv_torch(torch.diag(sl))
        usx = torch.mm( sl_inv,
                torch.mm(
                    torch.diag(s_diag), torch.mm(u.t(), x)
                )
            )
        return usx

    def lambda_fn(self,
                x: Tensor, 
                lambda_: Tensor, 
                u: Tensor, 
                s_diag: Tensor) -> torch.Tensor:

        usx = self.get_usx(x, lambda_, u, s_diag)
        objective = torch.mm(usx, usx.conj().t())
        objective = objective.sum()
        objective = L1torch(objective, torch.tensor(1.).to(x.get_device()))
        return objective


    def get_W_plus(self, 
            lambda_, 
            u, 
            s_diag, 
            v) -> Tensor:

        sl_diag = torch.pow(s_diag, 2.) + lambda_.to(u.get_device())
        sl_diag = torch.nan_to_num(sl_diag, nan=epsilon_, posinf=epsilon_, neginf=epsilon_)
        sl_inv = inv_torch(torch.diag(sl_diag))
        sl_inv = torch.nan_to_num(sl_inv, nan=epsilon_, posinf=epsilon_, neginf=epsilon_)
        S = torch.mm(sl_inv, torch.diag(s_diag))

        W = torch.mm(
                    torch.mm(torch.conj(u), S), v.t()
                ).t()
        return W



    def compute(self, 
                x_faltten: Tensor, 
                u: Tensor, 
                s_diag: Tensor, 
                v: Tensor, 
                lambda_set: list,
    ):

        def lbfgs_closure():
            lbfgs_optim.zero_grad()
            objective = self.lambda_fn(x_mat, lambda_, u_mat, s_diag_mat)
            objective.backward(retain_graph=True)
            return objective


        # initilize lambda
        lambda_init = [1e1, 1e2, 1e3]
        _lambda_not_found = False

        if len(lambda_set) > 3:
            lambda_init = [coef*torch.median(lambda_set).item() for coef in [0.001, 0.01, 0.1, 1., 10.]] 

        #rint(torch.median(lambda_set).item())
        lr_init = [1.]

        # get other inputs
        u_mat = u
        #u_mat.requires_grad = False

        x_mat = x_faltten
        if x_mat.ndim > 2:
            x_mat = x_mat.view(x_mat.size(0),-1)
        #x_mat.requires_grad = False

        s_diag_mat = s_diag
        #s_diag_mat.requires_grad = False

        # find optimal lambda through lbfgs optimoization
        loss_lbfgs = [100]
        history_lambda = [0]
        for lr_i in lr_init:
            for lambd_ini in lambda_init:
                lambda_ = torch.ones(1,) * lambd_ini
                lambda_.requires_grad = True
            
                if abs(loss_lbfgs[-1]) > self.pred_tolerance:
                
                    # get lbfgs optimizer -> it is available for single device right now!
                    lbfgs_optim = torch.optim.LBFGS(
                                        params=[lambda_], 
                                        lr=lr_i,
                                        **self.lbfgs_kwargs)

                    lbfgs_optim.step(lbfgs_closure)
                    est_err = self.lambda_fn(x_mat, 
                                    lambda_, 
                                    u_mat, 
                                    s_diag_mat,
                                    
                    ).item()
                    loss_lbfgs.append(est_err)
                    history_lambda.append(lambda_)

    
        lambda_bin, loss_bin = [], []
        for lam_, loss_ in zip(history_lambda, loss_lbfgs):
            if  0 < lam_ < 1e8:
                lambda_bin.append(lam_)
                loss_bin.append(loss_)

        if len(lambda_bin) == 0:
            _lambda_not_found = True
        else:
            min_loss_index = loss_bin.index(min(loss_bin))
            min_loss = loss_bin[min_loss_index]


        if (not _lambda_not_found) and (min_loss < 10. * self.pred_tolerance): 
            closed_form_sol_not_found = False
            lambda_hat = lambda_bin[min_loss_index]
            W_plus = self.get_W_plus(lambda_hat, u, s_diag, v)
        else:
            closed_form_sol_not_found = True
            lambda_hat = torch.zeros(1, dtype=u.dtype, requires_grad=True)
            W_plus = self.get_W_plus(lambda_hat, u, s_diag, v)

        return W_plus, lambda_hat, closed_form_sol_not_found


def _svd_decomposition(data, sigma_THR, sigma_MIN):
    """ Calculates SVD""" 
    if torchV >= 13:
        kwargs = {'full_matrices': False}
    else:
        kwargs = {}

    try:
        u, s_diag, vh = svd_torch(data, **kwargs)
        thr = torch.max(s_diag) * sigma_THR
        s_diag = torch.where(s_diag > thr.item(), s_diag, sigma_MIN)
        v = vh.mH if torchV >= 13 else vh #torch.adjoint(vh)
        return u, s_diag, v

    except:
        try: 
            # Fix the singularity problem partially
            u, s_diag, vh = svd_torch(data + 1e-4*data.mean()*torch.rand_like(data)) 
            thr = torch.max(s_diag) * sigma_THR
            s_diag = torch.where(s_diag > thr.item(), s_diag, sigma_MIN)
            v = vh.mH if torchV >= 13 else vh #torch.adjoint(vh)
            return u, s_diag, v
        except: 
            return None, None, None


def _get_norm_inp(normalize_input: bool, 
                num_channels: int, 
                layer_dim: List, 
                affine: bool):
    """ Returns normalisaion of inputs if required """
    
    if normalize_input:
        if len(layer_dim) < 2:
            bn = getattr(nn, "LayerNorm") 
            norm_ = bn([num_channels]+layer_dim, elementwise_affine=affine)
        else:
            # BatchNorm for 2d or higher dimension data
            bn = getattr(nn, "BatchNorm%dd"%len(layer_dim)) 
            norm_ = bn(num_channels, affine=affine)
        
        if affine:
            nn.init.constant_(norm_.weight, 1)
            nn.init.constant_(norm_.bias, 0)
    else:
        norm_ = nn.Identity() 
    return norm_

def _get_norm_out(normalize_input: bool, 
                num_channels: int, 
                layer_dim: List, 
                affine: bool):
    """ Returns normalisaion of outputs if required """

    if normalize_input:
        norm_ = nn.LayerNorm([num_channels]+layer_dim, elementwise_affine=affine) 
        if affine:
            nn.init.constant_(norm_.weight, 1)
            nn.init.constant_(norm_.bias, 0)

    else:
        norm_ = nn.Identity()  

    return norm_


    

if __name__=="__main__":
    gpu = 0

    batchSize = 100
    f = torch.rand([batchSize, 128]).to(f"cuda:{gpu}")
    g = torch.rand([batchSize, 16]).to(f"cuda:{gpu}")
    kwargs = {
                'gpu': gpu,
                'f_num_channels': 128, 
                'g_num_channels': 16,
                'f_layer_dim': [],
                'g_layer_dim':[],
                'normalize_input': True,
                'normalize_output': True,
                'affine': True,
                'sigma_THR': 0.0, 
                'sigma_MIN': 0.0, 
                }

    regbn_module = RegBN(**kwargs).to(f"cuda:{gpu}")
    print(regbn_module)

    kwargs_train = {"is_training": True, 'n_epoch': 1, 'steps_per_epoch': 100}
    f_n, g_n = regbn_module(f, g, **kwargs_train) 
    print(f.shape, f_n.shape)



# TODO: Add more examples