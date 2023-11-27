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

class FMAGI(object):
    def __init__(self, ys, dynamic, grid_size=201, interpolation_size = 201, interpolation_orders=3):
        self.grid_size = grid_size
        self.interpolation_size = interpolation_size
        self.comp_size = len(ys)
        for i in range(self.comp_size):
            if (not torch.is_tensor(ys[i])):
                ys[i] = torch.tensor(ys[i]).double().squeeze()
        self.ys = ys
        self.fOde = dynamic
        self._kiss_gp_initialization(interpolation_orders=interpolation_orders)

    def _kiss_gp_initialization(self, interpolation_orders=5, training_iterations=100):
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
        # print('grid',self.grid)
        # print('interp',self.interp)

    def map(self, max_epoch=1000, 
            learning_rate=1e-3, decay_learning_rate=True,
            robust=True, robust_eps=0.05, 
            hyperparams_update=True, dynamic_standardization=False,
            verbose=False, returnX=False,interpolation = True):
        gpmat = []
        lkh_1= []
        lkh_2 = []
        u = torch.empty(self.grid_size, self.comp_size).double()
        x = torch.empty(self.grid_size, self.comp_size).double()
        dxdtGP = torch.empty(self.grid_size, self.comp_size).double()

        with torch.no_grad():
            for i in range(self.comp_size):
                ti = self.ys[i][:,0]
                # print(ti.size(0))
                model = self.gp_models[i]
                mean = model.mean_module.constant.item()
                outputscale = model.covar_module.outputscale.item()
                noisescale = model.likelihood.noise.item()
                nugget = noisescale / outputscale
                grid_kernel = model.covar_module.base_kernel
                base_kernel = grid_kernel.base_kernel
                # compute mean for grid points
                xi = model(self.grid).mean
                # print(xi.size(0))
                LC = base_kernel(self.grid,self.grid).add_jitter(1e-6)._cholesky()
                LCinv = LC.inverse()
                ui = LCinv.matmul(xi-mean) / np.sqrt(outputscale)
                # compute uq for the grid points
                q = grid_kernel(ti,ti).add_jitter(nugget)._cholesky().inverse().matmul(grid_kernel(ti,self.grid))
                LU = (base_kernel(self.grid,self.grid)-q.t().matmul(q)).add_jitter(1e-6)._cholesky().mul(np.sqrt(outputscale))
                # compute gradient for grid points
                m = LCinv.matmul(base_kernel.dCdx2(self.grid,self.grid)).t()
                dxi = m.matmul(ui) * np.sqrt(outputscale)
                LKinv = (base_kernel.d2Cdx1dx2(self.grid,self.grid)-m.matmul(m.t())).add_jitter(1e-6)._cholesky().inverse()
                # print('LKinv',LKinv.item())
                m = m.matmul(LCinv)
                # print('m',m.item())
                # compute covariance for x|grid
                s = LCinv.matmul(grid_kernel(self.grid,ti))
                LQinv = (grid_kernel(ti,ti).add_jitter(nugget) - s.t().matmul(s)).add_jitter(1e-6)._cholesky().inverse()
                s = s.t().matmul(LCinv)
                u[:,i] = ui
                x[:,i] = xi
                print('u',u[1:10,i])
                # print('x',xi.size(0))
                dxdtGP[:,i] = dxi
                gpmat.append({'LC':LC,'LCinv':LCinv,'m':m,'LKinv':LKinv,'s':s,'LQinv':LQinv,'LU':LU})

        # interpolation 
        if (interpolation):
            gpimat = []
            grid_min = self.grid.min()
            grid_max = self.grid.max()
            self.interp = torch.linspace(grid_min, grid_max, self.interpolation_size)
            # print('interp',self.interp)
            xi = torch.empty(self.interp.size(0), self.comp_size).double()
            ui = torch.empty(self.interp.size(0), self.comp_size).double()
            with torch.no_grad():
                for i in range(self.comp_size):   
                    # compute gradient for grid points
                    model = self.gp_models[i]
                    mean = model.mean_module.constant.item()
                    outputscale = model.covar_module.outputscale.item()
                    xii = model(self.interp).mean
                    LCt = base_kernel(self.interp,self.interp).add_jitter(1e-6)._cholesky()
                    LCinvt = LCt.inverse()
                    uii = LCinvt.matmul(xii-mean) / np.sqrt(outputscale)
                    # print('uii',uii.size())
                    m = LCinv.matmul(base_kernel.dCdx2(self.interp,self.grid)).t()
                    # print('m',m.size())
                    LKinv = (base_kernel.d2Cdx1dx2(self.interp,self.interp)-m.matmul(m.t())).add_jitter(1e-6)._cholesky().inverse()
                    # print('LKinv',LKinv.item())
                    m = m.matmul(LCinv)
                    # print('m',m.item())
                    ui[:,i] = uii
                    # print('value',value[1:10,i])
                    print('ui',ui[1:10,i])
                    gpimat.append({'m':m,'LKinv':LKinv,'LC':LCt})
        
        # output standardization for fOde
        if (dynamic_standardization):
            dxdtGP_means = torch.mean(dxdtGP, axis=0)
            dxdtGP_stds = torch.std(dxdtGP, axis=0)
            self.fOde.update_output_layer(dxdtGP_means, dxdtGP_stds)


        # optimizer for u and theta
        state_optimizer = torch.optim.Adam([u], lr=learning_rate)
        inter_optimizer = torch.optim.Adam([ui], lr=learning_rate)
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
            theta_optimizer.step()

        for epoch in range(max_epoch):
            # optimize u (x after Cholesky decomposition)
            u.requires_grad_(True)
            for st in range(1):
                state_optimizer.zero_grad()
                # reconstruct x
                x = torch.empty_like(u).double()
                # print('u',x)
                for i in range(self.comp_size):
                    mean = self.gp_models[i].mean_module.constant.item()
                    outputscale = self.gp_models[i].covar_module.outputscale.item()
                    x[:,i] = mean + np.sqrt(outputscale) * gpmat[i]['LC'].matmul(u[:,i])
                    
                dxdtOde = self.fOde(x)
                print('interp',x[1:10,:])
                print('dxdtOde',dxdtOde[1:10,:])
                #####################################
                if (interpolation):
                    # interpolate dxdtOde
                    ui.requires_grad_(True)
                    for st in range(1):
                        inter_optimizer.zero_grad()
                        interp = torch.empty_like(ui).double()
                        for i in range(self.comp_size):
                            mean = self.gp_models[i].mean_module.constant.item()
                            outputscale = self.gp_models[i].covar_module.outputscale.item()
                            interp[:,i] = mean + np.sqrt(outputscale) * gpimat[i]['LC'].matmul(ui[:,i]) #using updated version to get dxdtodde 
                        dxdtOde_int = self.fOde(interp)
                    print('interp',interp[1:10,:])
                    print('dxdtOde_int',dxdtOde_int[1:10,:])
                #####################################
                lkh = torch.zeros((self.comp_size, 3))
                for i in range(self.comp_size):
                    mean = self.gp_models[i].mean_module.constant.item()
                    outputscale = self.gp_models[i].covar_module.outputscale.item()
                    # p(X[I] = x[I]) = P(U[I] = u[I])
                    lkh[i,0] = -0.5 * u[:,i].square().sum() #161 
                    # p(Y[I] = y[I] | X[I] = x[I])
                    yiError = gpmat[i]['LQinv'].matmul(self.ys[i][:,1]-(mean+gpmat[i]['s'].matmul(x[:,i]-mean))) #41
                    lkh[i,1] = -0.5/outputscale * yiError.square().sum()
                    # print('yiError',yiError.size(0))
                    # p(X'[I]=f(x[I],theta)|X(I)=x(I)) to p(X'[J]=f(x[J],theta)|X(I)=x(I)) by interpolation
                    if (interpolation):
                        dxidtError = gpmat[i]['LKinv'].matmul(dxdtOde_int[:,i]-gpmat[i]['m'].matmul(x[:,i]-mean))
                        print('dxidtError',dxidtError)
                        lkh[i,2] = -0.5/outputscale * dxidtError.square().sum() / self.interp.size(0) * yiError.size(0) #201
                    else:
                        dxidtError = gpmat[i]['LKinv'].matmul(dxdtOde[:,i]-gpmat[i]['m'].matmul(x[:,i]-mean))
                        print('dxidtError',dxidtError)
                        lkh[i,2] = -0.5/outputscale * dxidtError.square().sum() / self.grid_size * yiError.size(0)
                    lkh_2.append(lkh[i,2])
                state_loss = -torch.sum(lkh)  / self.grid_size
                state_loss.backward()
                state_optimizer.step()
            # detach gradient information
            u.requires_grad_(False)

            if (verbose and (epoch==0 or (epoch+1) % int(max_epoch/5) == 0)):
                print('%d/%d iteration: %.6f' %(epoch+1,max_epoch,state_loss.item()))

            # reconstruct x
            x = torch.empty_like(u).double()
            for i in range(self.comp_size):
                mean = self.gp_models[i].mean_module.constant.item()
                outputscale = self.gp_models[i].covar_module.outputscale.item()
                x[:,i] = mean + np.sqrt(outputscale) * gpmat[i]['LC'].matmul(u[:,i])

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
                    dxrdtOde = self.fOde(xr)
                    theta_optimizer.zero_grad()
                    lkh = torch.zeros(self.comp_size)
                    for i in range(self.comp_size):
                        mean = self.gp_models[i].mean_module.constant.item()
                        outputscale = self.gp_models[i].covar_module.outputscale.item()
                        dxrdtError = gpimat[i]['LKinv'].matmul(dxrdtOde[:,i]-gpimat[i]['m'].matmul(xr[:,i]-mean))
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
            return (self.grid.numpy(), x.numpy(), lkh_1, lkh_2)

    def predict(self, x0, ts, **params):
        # obtain prediction by numerical integration
        itg = intergrator.RungeKutta(self.fOde)
        ts = torch.tensor(ts).double().squeeze()
        xs = itg.forward(x0, ts, **params)
        return (xs.numpy())


