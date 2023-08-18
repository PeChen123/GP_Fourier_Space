import numpy as np
import torch
from scipy.special import kv, gamma
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import scipy.stats as st
import numpy.fft as fft
import scipy.special as fun
from scipy.stats import norm
import torch.optim as optim

'''Step one '''
# We first set the Matern kernel with nu = 2.01
# when the noise level is unknown;
# this part will provide us the phi_1 and phi_2(when noise level is known, otherwise,we can have the noise level) for observed comp


torch.set_default_dtype(torch.double)

class Bessel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, nu):
        ctx._nu = nu
        ctx.save_for_backward(inp)
        return torch.tensor(fun.kv(nu, inp.detach().numpy()))

    @staticmethod
    def backward(ctx, grad_out):
        inp, = ctx.saved_tensors
        nu = ctx._nu
        grad_in = torch.tensor(grad_out.numpy() * fun.kvp(nu, inp.detach().numpy()))
        return grad_in, None

class Matern:
    def __init__(self, nu=2.01, lengthscale=1e-1):
        self.nu = nu
        self.log_lengthscale = torch.tensor(np.log(lengthscale), requires_grad=True)

    @property
    def lengthscale(self):
        return torch.exp(self.log_lengthscale).item()

    def compute_kernel(self, x1, x2=None):
        lengthscale = torch.exp(self.log_lengthscale)
        if x2 is None:
            x2 = x1

        r = torch.abs(x1[:, None] - x2) / lengthscale
        r_scaled = torch.sqrt(2. * self.nu) * r
        r_scaled = r_scaled.clamp_min(1e-15)
        
        kernel = (2 ** (1 - self.nu) / fun.gamma(self.nu)) * (r_scaled ** self.nu)
        kernel *= Bessel.apply(r_scaled, self.nu)
        return kernel

    def kernel(self, x1, x2=None):
        return self.compute_kernel(x1, x2).detach()

    def dx1(self, x1, x2=None):
        lengthscale = torch.exp(self.log_lengthscale)
        x1 = x1.squeeze()
        if x2 is None:
            x2 = x1

        with torch.no_grad():
            kernel = self.kernel(x1, x2)
            dist = x1[:, None] - x2
            r = dist.abs()
            r_scaled = torch.sqrt(2. * self.nu) * r / lengthscale
            r_scaled = r_scaled.clamp_min(1e-15)

            d_kernel = kernel * (self.nu / r_scaled + fun.kvp(self.nu, r_scaled) / fun.kv(self.nu, r_scaled))
            d_kernel *= torch.sqrt(2 * self.nu) / lengthscale
            d_kernel *= torch.sign(dist)
        return d_kernel

    def dx2(self, x1, x2=None):
        return -self.kernel_derivative_wrt_x1(x1, x2)

    def dx1x2(self, x1, x2=None):
        lengthscale = torch.exp(self.log_lengthscale)
        x1 = x1.squeeze()
        if x2 is None:
            x2 = x1

        with torch.no_grad():
            kernel = self.kernel(x1, x2)
            dist = x1[:, None] - x2
            r = dist.abs()
            r_scaled = torch.sqrt(2. * self.nu) * r / lengthscale
            r_scaled = r_scaled.clamp_min(1e-15)

            part1 = self.nu * (self.nu - 1) / r_scaled.square()
            part2 = 2. * self.nu / r_scaled * fun.kvp(self.nu, r_scaled) / fun.kv(self.nu, r_scaled)
            part3 = fun.kvp(self.nu, r_scaled, n=2) / fun.kv(self.nu, r_scaled)

            d2_kernel = kernel * (part1 + part2 + part3)
            d2_kernel[dist == 0] = -0.5 / (self.nu - 1)
            d2_kernel *= 2. * self.nu / lengthscale.square()
            d2_kernel = -d2_kernel
        return d2_kernel
    

# Kernel Matrix Computation
def compute_kernel_matrix(yI0, phi1, phi2):
    matern = Matern(nu=2.01, lengthscale=phi2)
    yI0_tensor = torch.tensor(yI0)
    K = matern.C(yI0_tensor).numpy() * phi1
    return K

# Fourier Transform for mu_phi2 computation
def compute_mu_phi2(yI0_data, I0):
    ft = fft.fft(yI0_data)
    freq = fft.fftfreq(len(I0), d=(I0[1]-I0[0]))
    pos_i = freq > 0
    pfreq = freq[pos_i]
    pw = np.abs(ft[pos_i])**2
    avef = np.dot(pfreq, pw) / np.sum(pw)
    mu_phi2 = 0.5 / avef
    return mu_phi2

# Priors
def prior_phi1(phi1):
    return 1  # Flat prior

def prior_phi2(phi2, mu_phi2, SD_phi2):
    return norm.pdf(phi2, loc=mu_phi2, scale=SD_phi2)  # Gaussian prior

def prior_sigma(sigma2):
    return 1  # Flat prior

# Gaussian likelihood
def p_mvn(y, mean, cov_matrix):
    n = y.shape[0]
    term1 = -0.5 * n * torch.log(torch.tensor(2.0 * np.pi))
    term2 = -0.5 * torch.logdet(cov_matrix)
    term3 = -0.5 * (y - mean).T @ torch.inverse(cov_matrix) @ (y - mean)
    return term1 + term2 + term3

# Likelihood
def likelihood(params, yI0):
    phi1, phi2, sigma2 = params
    K_phi = compute_kernel_matrix(yI0, phi1, phi2)
    cov_matrix = torch.tensor(K_phi + sigma2 * np.eye(len(yI0)))
    return p_mvn(torch.tensor(yI0), torch.zeros_like(torch.tensor(yI0)), cov_matrix)

# Objective function for optimization
def objective(params, yI0, mu_phi2, SD_phi2):
    phi1, phi2, sigma2 = params
    log_prior_phi1 = np.log(prior_phi1(phi1))
    log_prior_phi2 = np.log(prior_phi2(phi2, mu_phi2, SD_phi2))
    log_prior_sigma = np.log(prior_sigma(sigma2))
    log_joint_prior = log_prior_phi1 + log_prior_phi2 + log_prior_sigma
    log_likelihood_val = likelihood(params, yI0)
    return -(log_joint_prior + log_likelihood_val.item())

# run something like minimize(objective, initial_guess, args=(yI0_data, mu_phi2, SD_phi2))
# we can get phi1_opt, phi2_opt, sigma2_opt 

'''Step 2 initialization'''
# In this step, we are going to find the XI and start parameter

def initialize_XI(tau, Y_tau, I):
    tau = []
    y_tau = []
    interpolation = interp1d(tau, Y_tau, kind='linear', fill_value="extrapolate")
    XI = interpolation(I)
    return XI

# data
# tau is time point set
# Y_tau = value of tau
# I is observation time points (s.t tau in I)
# XI 

def U(theta, I, D, xd, y, tau, fx, N, sigma):
    matern = Matern()
    
    Cd = matern.kernel(I) 
    inv1 = torch.inverse(Cd)
    md = torch.mm(matern.dx1(I), inv1)
    K1 = matern.dx1x2(I)
    K3 = matern.dx2(I)
    Kd = torch.sub(K1, torch.mm(md, K3))

    potential = 0  # Initialize the potential energy
    
    for d in range(1, D+1):
        # 
        term1 = len(I) * np.log(2 * np.pi)
        term2 = np.log(np.linalg.det(Cd))
        term3 = np.linalg.norm(xd[d](I) - md[d](I))**2 * np.linalg.inv(Cd)
        
        term4 = len(I) * np.log(2 * np.pi)
        term5 = np.log(np.linalg.det(Kd))
        #  fx is some functions that maps theta and xd to a value
        term6 = np.linalg.norm(fx(theta, xd[d](I)) - md[d](I))**2 * np.linalg.inv(Kd)
        
        term7 = N[d] * np.log(2 * np.pi * sigma[d])
        term8 = np.linalg.norm(xd[d](tau[d]) - y[d](tau[d]))**2 / sigma[d]
        
        potential += term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8
    
    return -potential

def optimize_theta(initial_theta, I, D, xd, y, tau, fx, N, sigma):
    num_iter = []
    theta = initial_theta.clone()
    theta.requires_grad_(True)

    optimizer = optim.Adam([theta], lr=0.01)  # Using Adam optimizer as an example

    # Define the optimization loop
    for _ in range(num_iter):  # 
        optimizer.zero_grad()
        potential = U(theta, I, D, xd, y, tau, fx, N, sigma)
        potential.backward()
        optimizer.step()

    return theta.detach()

''' HMC sampler step '''
# U with tempering in sampling procedure
def Ut(theta, I, D, xd, y, tau, fx, N_d, sigma):
    matern = Matern()
    beta = D * len(I) / sum(N_d)
    Cd = matern.kernel(I) 
    inv1 = torch.inverse(Cd)
    md = torch.mm(matern.dx1(I), inv1)
    K1 = matern.dx1x2(I)
    K3 = matern.dx2(I)
    Kd = torch.sub(K1, torch.mm(md, K3))

    potential = 0  # Initialize the potential energy
    
    for d in range(1, D+1):
        #
        term1 = len(I) * np.log(2 * np.pi)
        term2 = np.log(np.linalg.det(Cd))
        term3 = np.linalg.norm(xd[d](I) - md[d](I))**2 * np.linalg.inv(Cd)
        
        term4 = len(I) * np.log(2 * np.pi)
        term5 = np.log(np.linalg.det(Kd))
        #  fx is some functions that maps theta and xd to a value
        term6 = np.linalg.norm(fx(theta, xd[d](I)) - md[d](I))**2 * np.linalg.inv(Kd)
        
        term7 = N[d] * np.log(2 * np.pi * sigma[d])
        term8 = np.linalg.norm(xd[d](tau[d]) - y[d](tau[d]))**2 / sigma[d]
        
        potential += term1 + term2 + term4 + term5  + term7 + term8 + 1/beta(term3 + term6)
    
    return -potential

# step 2 kinetic energy 
def K(p):
    ke = 0.5 * torch.dot(p, p)
    return ke

# step 3 Hamiltonian
def H(q, p, I, D, Cd, xd, mud, Kd, fx, md, N, sigma, tau, y,):
    Ha = Ut(q, I, D, Cd, xd, mud, Kd, fx, md, N, sigma, tau, y) + K(p)
    joint = torch.exp(Ha)
    return Ha, joint
# iter

# Gradient
def U_grad(q):
    q.requires_grad_(True)  
    potential = Ut(q)  
    potential.backward()  
    return q.grad 
    
# leapfrog
def leapfrog(U_grad, q, p, step_size, steps):
    q_new = q.clone()
    p_new = p.clone()
    
    for _ in range(steps):
        # Half-step update for momentum
        p_new -= step_size / 2 * U_grad(q_new)
        
        # Full-step update for position
        q_new += step_size * p_new
        
        # Another half-step update for momentum
        p_new -= step_size / 2 * U_grad(q_new)
    
    return q_new, p_new


def hmc_iteration(U_grad, K, q_current, leapfrog_steps, step_size,):

    # 1. Sample momentum
    p_current = torch.randn(len(q_current))
    
    # 2. Leapfrog integration to get proposed state (q*, p*)
    q_proposed, p_proposed = leapfrog(U_grad, q_current, p_current, leapfrog_steps, step_size)
    
    # 3. Metropolis acceptance criterion
    current_energy = Ut(q_current) + K(p_current)
    proposed_energy = Ut(q_proposed) + K(p_proposed)
    
    acceptance_ratio = torch.exp(current_energy - proposed_energy)
    if torch.rand(1) < acceptance_ratio:
        q_next,p_next = q_proposed,p_proposed
    else:
        q_next, p_next = q_current, p_current

    return q_next,p_next 