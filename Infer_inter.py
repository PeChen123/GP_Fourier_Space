import torch
import gpytorch
import numpy as np
import torch.fft as ft

from Kernal import MaternKernel
import intergrator
from grid_interpolation import GridInterpolationKernel
from linear_operator import to_linear_operator 
torch.set_default_dtype(torch.double)

class KISSGPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, grid, interpolation_orders):
        super(KISSGPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            GridInterpolationKernel(
                MaternKernel(), grid=grid, interpolation_orders=interpolation_orders
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class FTMAGI(object):
    def __init__(self, ys, dynamic, grid_size=201,interpolation_size = 201, interpolation_orders=3):
        self.grid_size = grid_size
        self.interpolation_size = interpolation_size
        self.comp_size = len(ys)
        for i in range(self.comp_size):
            if (not torch.is_tensor(ys[i])):
                ys[i] = torch.tensor(ys[i]).double().squeeze()
        self.ys = ys
        self.fOde = dynamic
        self._kiss_gp_initialization(interpolation_orders=interpolation_orders)

    def _kiss_gp_initialization(self, interpolation_orders=3, training_iterations=100):
        tmin = self.ys[0][:,0].min()
        tmax = self.ys[0][:,0].max()
        for i in range(1, self.comp_size):
            tmin = torch.min(tmin, self.ys[i][:,0].min())
            tmax = torch.max(tmax, self.ys[i][:,0].max())
        spacing = (tmax - tmin) / (self.grid_size - 1)
        padding = int((interpolation_orders + 1) / 2)
        grid_bounds = (tmin - padding * spacing, tmax + padding * spacing)
        self.grid = torch.linspace(grid_bounds[0], grid_bounds[1], self.grid_size+2*padding)
        self.gp_models = []
        for i in range(self.comp_size):
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = KISSGPRegressionModel(self.ys[i][:,0], self.ys[i][:,1], 
                likelihood, self.grid, interpolation_orders)
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model) # loss function
            for j in range(training_iterations):
                optimizer.zero_grad()
                output = model(self.ys[i][:,0])
                loss = -mll(output, self.ys[i][:,1])
                loss.backward()
                optimizer.step()
            model.eval()
            likelihood.eval()
            self.gp_models.append(model)
        self.grid = self.grid[padding:-padding] # remove extended grid points

    def map(self, max_epoch=1000, 
            learning_rate=1e-3, decay_learning_rate=True,
            robust=True, robust_eps=0.05, 
            hyperparams_update=True, dynamic_standardization=False,
            verbose=False, returnX=False,Truncated=False,k = None,interpolation = True):
        gpmat = []
        u = torch.empty(self.grid_size, self.comp_size).double()
        x = torch.empty(self.grid_size, self.comp_size).double()
        dxdtGP = torch.empty(self.grid_size, self.comp_size).double()
        grid_min = self.grid.min()
        grid_max = self.grid.max()
        self.interp = torch.linspace(grid_min, grid_max, self.interpolation_size)
        x_i = torch.empty(self.interp.size(0), self.comp_size).double()
        u_i = torch.empty(self.interp.size(0), self.comp_size).double()

        if Truncated:
            if k is None: # default k
                if self.grid_size % 2 == 1:
                    k = (self.grid_size + 1) // 2
                else:
                    k = self.grid_size // 2
            else: 
                k = k

        with torch.no_grad():
            for i in range(self.comp_size):
                ti = self.ys[i][:,0]
                model = self.gp_models[i]
                mean = model.mean_module.constant.item()
                outputscale = model.covar_module.outputscale.item()
                noisescale = model.likelihood.noise.item()
                nugget = noisescale / outputscale
                grid_kernel = model.covar_module.base_kernel
                base_kernel = grid_kernel.base_kernel
                # compute mean for grid points
                xi = model(self.grid).mean
                LC = base_kernel(self.grid,self.grid).add_jitter(1e-6)._cholesky()
                LCinv = LC.inverse()
                ui = LCinv.matmul(xi-mean) / np.sqrt(outputscale)
                # compute uq for the grid points
                q = grid_kernel(ti,ti).add_jitter(nugget)._cholesky().inverse().matmul(grid_kernel(ti,self.grid))
                LU = (base_kernel(self.grid,self.grid)-q.t().matmul(q)).add_jitter(1e-6)._cholesky().mul(np.sqrt(outputscale))
                # compute gradient for grid points
                m = LCinv.matmul(base_kernel.dCdx2(self.grid,self.grid)).t()
                dxi = m.matmul(ui) * np.sqrt(outputscale)
                ##############################################
                LK = (base_kernel.d2Cdx1dx2(self.grid,self.grid)-m.matmul(m.t())).to_dense()
                pre = ft.fft(LK)
                pre1 = torch.fft.fft(pre.t().conj()).real
                pre2 = torch.fft.fft(pre.t()).real
                pre3 = torch.fft.fft(pre.t().conj()).imag
                pre4 = torch.fft.fft(pre.t()).imag
                final1t = 0.5 * (pre1 + pre2)[:k, :k]         # top left
                final2t = 0.5 * (pre1 - pre2)[1:k, 1:k]       # bottom right
                final3t = 0.5 * (pre3 + pre4)[:k, 1:k]       # top right
                final4t = -0.5 * (pre3 - pre4)[1:k, :k]
                truncated_matrix = torch.cat((torch.cat((final1t, final4t), 0), 
                                            torch.cat((final3t, final2t), 0)), 1)
                LK = to_linear_operator(truncated_matrix)
                LKinv = LK.add_jitter(1e-6)._cholesky().inverse()
                ##############################################
                m = m.matmul(LCinv)
                # compute covariance for x|grid
                s = LCinv.matmul(grid_kernel(self.grid,ti))
                LQinv = (grid_kernel(ti,ti).add_jitter(nugget) - s.t().matmul(s)).add_jitter(1e-6)._cholesky().inverse()
                s = s.t().matmul(LCinv)
                m_int = LCinv.matmul(base_kernel.dCdx2(self.interp,self.grid)).t()
                LKinv_int = (base_kernel.d2Cdx1dx2(self.interp,self.interp) - m_int.matmul(m_int.t())).to_dense()
                pre_int = ft.fft(LKinv_int)
                pre1_int = torch.fft.fft(pre_int.t().conj()).real
                pre2_int = torch.fft.fft(pre_int.t()).real
                pre3_int = torch.fft.fft(pre_int.t().conj()).imag
                pre4_int = torch.fft.fft(pre_int.t()).imag
                final1t_int = 0.5 * (pre1_int + pre2_int)[:k, :k]         # top left
                final2t_int = 0.5 * (pre1_int - pre2_int)[1:k, 1:k]       # bottom right
                final3t_int = 0.5 * (pre3_int + pre4_int)[:k, 1:k]       # top right
                final4t_int = -0.5 * (pre3_int - pre4_int)[1:k, :k]
                truncated_matrix_int = torch.cat((torch.cat((final1t_int, final4t_int), 0), 
                                            torch.cat((final3t_int, final2t_int), 0)), 1)
                LKinv_int = to_linear_operator(truncated_matrix_int)
                LKinv_int = LKinv_int.add_jitter(1e-6)._cholesky().inverse()
                m_int = m_int.matmul(LCinv)
                # print('m_int',m_int.size())
                # print('LKinv_int',LKinv_int.size())
                xii = model(self.interp).mean
                LCt = base_kernel(self.interp,self.interp).add_jitter(1e-6)._cholesky()
                LCinvt = LCt.inverse()
                uii = LCinvt.matmul(xii-mean) / np.sqrt(outputscale)
                u_i[:,i] = uii
                x_i[:,i] = xii
                # store information
                u[:,i] = ui
                x[:,i] = xi
                dxdtGP[:,i] = dxi
                gpmat.append({'LC':LC,'LCinv':LCinv,'m':m,'LKinv':LKinv,'s':s,'LQinv':LQinv,'LU':LU,'m_int':m_int,'LKinv_int':LKinv_int,'LCt':LCt,'LCinvt':LCinvt})

        # output standardization for fOde
        if (dynamic_standardization):
            dxdtGP_means = torch.mean(dxdtGP, axis=0)
            dxdtGP_stds = torch.std(dxdtGP, axis=0)
            self.fOde.update_output_layer(dxdtGP_means, dxdtGP_stds)
        

        # optimizer for u and theta
        state_optimizer = torch.optim.Adam([u], lr=learning_rate)
        inter_optimizer = torch.optim.Adam([u_i], lr=learning_rate)
        theta_optimizer = torch.optim.Adam(self.fOde.parameters(), lr=learning_rate)
        theta_lambda = lambda epoch: (epoch+1) ** (-0.5)
        theta_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(theta_optimizer, lr_lambda=theta_lambda)

        # optimize initial theta
        # attach gradient for theta
        eps = robust_eps
        for param in self.fOde.parameters():
            param.requires_grad_(True)
        for tt in range(200):
            if (robust):
                # identify adversarial path
                delta = torch.zeros_like(x, requires_grad=True)
                xr = torch.empty_like(x)
                for i in range(self.comp_size):
                    xr[:,i] = x[:,i] + gpmat[i]['LU'].matmul(delta[:,i])
                dxrdtOde = self.fOde(xr)
                theta_optimizer.zero_grad()
                lkh = torch.zeros(self.comp_size)
                for i in range(self.comp_size):
                    mean = self.gp_models[i].mean_module.constant.item()
                    outputscale = self.gp_models[i].covar_module.outputscale.item()
                    dxrdtError = gpmat[i]['LKinv'].matmul(dxrdtOde[:,i]-gpmat[i]['m'].matmul(xr[:,i]-mean))
                    lkh[i] = -0.5/outputscale * dxrdtError.square().mean()
                theta_loss = -torch.sum(lkh)
                theta_loss.backward()
                xr = torch.empty_like(x)
                for i in range(self.comp_size):
                    # gradient ascent to find adversarial path
                    xr[:,i] = x[:,i] + gpmat[i]['LU'].matmul(eps*delta.grad.data.sign()[:,i])
            else:
                xr = x.clone()
            if (interpolation):
                interp = x_i.clone()
                dxrdtOde = self.fOde(interp)
            else:
                dxrdtOde = self.fOde(xr)
            theta_optimizer.zero_grad()
            lkh = torch.zeros(self.comp_size)
            for i in range(self.comp_size):
                mean = self.gp_models[i].mean_module.constant.item()
                outputscale = self.gp_models[i].covar_module.outputscale.item()
                ##############################################
                xr_ft = ft.fft(gpmat[i]['m_int'].matmul(xr[:,i]-mean))[:k]
                rp,ip = xr_ft.real,xr_ft.imag[1:k]
                xr_ft = torch.cat((rp,ip),0)

                dxrdtOde_ft = ft.fft(dxrdtOde[:,i])[:k]
                rp,ip = dxrdtOde_ft.real,dxrdtOde_ft.imag[1:k]
                dxrdtOde_ft = torch.cat((rp,ip),0)

                dxrdtError = gpmat[i]['LKinv_int'].matmul(dxrdtOde_ft-xr_ft)
                ##############################################
                lkh[i] = -0.5/outputscale * dxrdtError.square().mean()
            theta_loss = -torch.sum(lkh)
            theta_loss.backward(retain_graph=True)
            theta_optimizer.step()

        # detach theta gradient
        for param in self.fOde.parameters():
            param.requires_grad_(False)

        for epoch in range(max_epoch):
            # optimize u (x after Cholesky decomposition)
            u.requires_grad_(True)
            u_i.requires_grad_(True)
            for st in range(1):
                state_optimizer.zero_grad()
                inter_optimizer.zero_grad()
                # reconstruct x
                x = torch.empty_like(u).double()
                x_i = torch.empty_like(u_i).double()
                for i in range(self.comp_size):
                    mean = self.gp_models[i].mean_module.constant.item()
                    outputscale = self.gp_models[i].covar_module.outputscale.item()
                    x[:,i] = mean + np.sqrt(outputscale) * gpmat[i]['LC'].matmul(u[:,i])
                    x_i[:,i] = mean + np.sqrt(outputscale) * gpmat[i]['LCt'].matmul(u_i[:,i])
                dxdtOde = self.fOde(x)
                dxdtOde_int = self.fOde(x_i)
                lkh = torch.zeros((self.comp_size, 3))
                for i in range(self.comp_size):
                    mean = self.gp_models[i].mean_module.constant.item()
                    outputscale = self.gp_models[i].covar_module.outputscale.item()
                    # p(X[I] = x[I]) = P(U[I] = u[I])
                    lkh[i,0] = -0.5 * u[:,i].square().sum()
                    # p(Y[I] = y[I] | X[I] = x[I])
                    yiError = gpmat[i]['LQinv'].matmul(self.ys[i][:,1]-(mean+gpmat[i]['s'].matmul(x[:,i]-mean)))
                    lkh[i,1] = -0.5/outputscale * yiError.square().sum()
                    # p(X'[I]=f(x[I],theta)|X(I)=x(I))
                    ##############################################
                    if (interpolation):
                        dxdtOde_int_ft = ft.fft(dxdtOde_int[:,i])[:k]
                        rp1,ip1 = dxdtOde_int_ft.real, dxdtOde_int_ft.imag[1:k]
                        dxdtOde_int_ft= torch.cat((rp1,ip1),0)
                        x_int_ft = ft.fft(gpmat[i]['m_int'].matmul(x[:,i]-mean))[:k]
                        rp2,ip2= x_int_ft.real, x_int_ft.imag[1:k]
                        x_int_ft= torch.cat((rp2,ip2),0)
                        dxidtError = gpmat[i]['LKinv_int'].matmul(dxdtOde_int_ft-x_int_ft)
                        lkh[i,2] = -0.5/outputscale * dxidtError.square().sum() / self.interp.size(0) * yiError.size(0)
                    else:
                        dxdtOde_ft= ft.fft(dxdtOde[:,i])[:k]
                        rp1,ip1 = dxdtOde_ft.real, dxdtOde_ft.imag[1:k]
                        dxdtOde_ft= torch.cat((rp1,ip1),0)

                        x_ft = ft.fft(gpmat[i]['m'].matmul(x[:,i]-mean))[:k]
                        rp2,ip2= x_ft.real, x_ft.imag[1:k]
                        x_ft= torch.cat((rp2,ip2),0)

                        dxidtError = gpmat[i]['LKinv'].matmul(dxdtOde_ft-x_ft)
                        ##############################################
                        lkh[i,2] = -0.5/outputscale * dxidtError.square().sum() / self.grid_size * yiError.size(0)

                state_loss = -torch.sum(lkh)  / self.grid_size
                state_loss.backward(retain_graph=True)
                state_optimizer.step()
                inter_optimizer.step()
            # detach gradient information
            u.requires_grad_(False)
            u_i.requires_grad_(False)

            if (verbose and (epoch==0 or (epoch+1) % int(max_epoch/5) == 0)):
                print('%d/%d iteration: %.6f' %(epoch+1,max_epoch,state_loss.item()))

            # reconstruct x
            x = torch.empty_like(u).double()
            x_i = torch.empty_like(u_i).double()
            for i in range(self.comp_size):
                mean = self.gp_models[i].mean_module.constant.item()
                outputscale = self.gp_models[i].covar_module.outputscale.item()
                x[:,i] = mean + np.sqrt(outputscale) * gpmat[i]['LC'].matmul(u[:,i])
                x_i[:,i] = mean + np.sqrt(outputscale) * gpmat[i]['LCt'].matmul(u_i[:,i])

            if ((epoch+1) < max_epoch):
                # update hyperparameter
                if (((epoch+1) % int(max_epoch/5) == 0) and hyperparams_update):
                    dxdtOde = self.fOde(x)
                    for i in range(self.comp_size):
                        ti = self.ys[i][:,0]
                        yi = self.ys[i][:,1]
                        xi = x[:,i]
                        model = self.gp_models[i]
                        model.train() 
                        model.likelihood.train()
                        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
                        for j in range(5):
                            optimizer.zero_grad()
                            # p(X[I] = x[I]) 
                            LC = model.covar_module.base_kernel.base_kernel(self.grid,self.grid)._cholesky()
                            LCinv = LC.inverse()
                            xiError = LCinv.matmul(xi-model.mean_module.constant)
                            lkh1 = -0.5 / model.covar_module.outputscale * xiError.square().sum()
                            lkh1 = lkh1 - 0.5 * self.grid_size * model.covar_module.outputscale.log() - LC.logdet()
                            # p(Y[I] = y[I] | X[I] = x[I])
                            nugget = model.covar_module.outputscale / model.likelihood.noise
                            s = LCinv.matmul(model.covar_module.base_kernel(self.grid,ti))
                            LQ = (model.covar_module.base_kernel(ti,ti).add_diagonal(nugget) - s.t().matmul(s))._cholesky()
                            s = s.t().matmul(LCinv)
                            yiError = LQ.inverse().matmul(yi-(model.mean_module.constant+s.matmul(xi-model.mean_module.constant)))
                            lkh2 = -0.5 / model.covar_module.outputscale * yiError.square().sum()
                            lkh2 = lkh2 - 0.5 * self.grid_size * model.covar_module.outputscale.log() - LQ.logdet()
                            # p(X'[I]=f(x[I],theta)|X(I)=x(I))
                            m = LCinv.matmul(model.covar_module.base_kernel.base_kernel.dCdx2(self.grid,self.grid)).t()
                            LK = (model.covar_module.base_kernel.base_kernel.d2Cdx1dx2(self.grid,self.grid)-m.matmul(m.t())).add_jitter(1e-6)._cholesky()
                            m = m.matmul(LCinv)
                            dxidtError = LK.inverse().matmul(dxdtOde[:,i]-m.matmul(x[:,i]-model.mean_module.constant))
                            lkh3 = - 0.5 / model.covar_module.outputscale * dxidtError.square().sum()
                            lkh3 = lkh3 - 0.5 * self.grid_size * model.covar_module.outputscale.log() - LK.logdet()
                            # loss = -(lkh1/self.grid_size + lkh2/ti.size(0) + lkh3/self.grid_size)
                            loss = -(lkh1 + lkh2 + lkh3/self.grid_size*ti.size(0)) / self.grid_size
                            loss.backward()
                            optimizer.step()
                        model.eval()
                        model.likelihood.eval()
                        self.gp_models[i] = model
                        # update gpmat
                        with torch.no_grad():
                            mean = model.mean_module.constant.item()
                            outputscale = model.covar_module.outputscale.item()
                            noisescale = model.likelihood.noise.item()
                            nugget = noisescale / outputscale
                            grid_kernel = model.covar_module.base_kernel
                            base_kernel = grid_kernel.base_kernel
                            # compute mean for the grid points
                            LC = base_kernel(self.grid,self.grid).add_jitter(1e-6)._cholesky()
                            LCinv = LC.inverse()
                            ui = LCinv.matmul(xi-mean) / np.sqrt(outputscale)
                            # compute uq for the grid points
                            q = grid_kernel(ti,ti).add_jitter(nugget)._cholesky().inverse().matmul(grid_kernel(ti,self.grid))
                            LU = (base_kernel(self.grid,self.grid)-q.t().matmul(q)).add_jitter(1e-6)._cholesky().mul(np.sqrt(outputscale))
                            # compute gradient for grid points
                            m = LCinv.matmul(base_kernel.dCdx2(self.grid,self.grid)).t()
                            LKinv = (base_kernel.d2Cdx1dx2(self.grid,self.grid)-m.matmul(m.t())).add_jitter(1e-6)._cholesky().inverse()
                            m = m.matmul(LCinv)
                            # assuming fixed noise, compute covariance for x|grid
                            s = LCinv.matmul(grid_kernel(self.grid,ti))
                            LQinv = (grid_kernel(ti,ti).add_jitter(nugget) - s.t().matmul(s)).add_jitter(1e-6)._cholesky().inverse()
                            s = s.t().matmul(LCinv)
                            # store information
                            u[:,i] = ui
                            gpmat[i] = {'LC':LC,'LCinv':LCinv,'m':m,'LKinv':LKinv,'s':s,'LQinv':LQinv,'LU':LU}

                eps = robust_eps * ((epoch+1) ** (-0.5))
                for param in self.fOde.parameters():
                    param.requires_grad_(True)
                for tt in range(1):
                    if (robust):
                        # identify adversarial path
                        delta = torch.zeros_like(x, requires_grad=True)
                        xr = torch.empty_like(x)
                        for i in range(self.comp_size):
                            xr[:,i] = x[:,i] + gpmat[i]['LU'].matmul(delta[:,i])
                        dxrdtOde = self.fOde(xr)
                        theta_optimizer.zero_grad()
                        lkh = torch.zeros(self.comp_size)
                        for i in range(self.comp_size):
                            mean = self.gp_models[i].mean_module.constant.item()
                            outputscale = self.gp_models[i].covar_module.outputscale.item()
                            dxrdtError = gpmat[i]['LKinv'].matmul(dxrdtOde[:,i]-gpmat[i]['m'].matmul(xr[:,i]-mean))
                            lkh[i] = -0.5/outputscale * dxrdtError.square().mean()
                        theta_loss = -torch.sum(lkh)
                        theta_loss.backward()
                        xr = torch.empty_like(x)
                        for i in range(self.comp_size):
                            # gradient ascent to find adversarial path
                            xr[:,i] = x[:,i] + gpmat[i]['LU'].matmul(eps*delta.grad.data.sign()[:,i])
                    else:
                        xr = x.clone()
                    if (interpolation):
                        interp = x_i.clone()
                        dxrdtOde = self.fOde(interp)
                    else:
                        dxrdtOde = self.fOde(xr)
                    # optimze theta over the adversarial path
                    theta_optimizer.zero_grad()
                    lkh = torch.zeros(self.comp_size)
                    for i in range(self.comp_size):
                        mean = self.gp_models[i].mean_module.constant.item()
                        outputscale = self.gp_models[i].covar_module.outputscale.item()
                        ##############################################
                        xr_ft = ft.fft(gpmat[i]['m_int'].matmul(xr[:,i]-mean))[:k]
                        rp,ip = xr_ft.real,xr_ft.imag[1:k]
                        xr_ft = torch.cat((rp,ip),0)

                        dxrdtOde_ft = ft.fft(dxrdtOde[:,i])[:k]
                        rp,ip = dxrdtOde_ft.real,dxrdtOde_ft.imag[1:k]
                        dxrdtOde_ft = torch.cat((rp,ip),0)

                        dxrdtError = gpmat[i]['LKinv_int'].matmul(dxrdtOde_ft-xr_ft)
                        ##############################################
                        lkh[i] = -0.5/outputscale * dxrdtError.square().mean()
                    theta_loss = -torch.sum(lkh)
                    theta_loss.backward()
                    theta_optimizer.step()
                if (decay_learning_rate):
                    theta_lr_scheduler.step()
                # detach theta gradient
                for param in self.fOde.parameters():
                    param.requires_grad_(False)

        if (returnX):
            return (self.grid.numpy(), x.numpy())

    def predict(self, x0, ts, **params):
        # obtain prediction by numerical integration
        itg = intergrator.RungeKutta(self.fOde)
        ts = torch.tensor(ts).double().squeeze()
        xs = itg.forward(x0, ts, **params)
        return (xs.numpy())