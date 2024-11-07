import numpy as np
import matplotlib.pyplot as plt
import time
from numba.experimental import jitclass
from numba import njit

@njit
def d_s_arr(rho_arr):
    alpha = np.pi/2 - 1
    return (1-rho_arr)*(1-alpha*rho_arr + ((alpha*(2*alpha-1))/(2*alpha+1))*rho_arr**2 )
    
def D(rho):
    return (1-d_s_arr(rho))/rho

@njit
def s_arr(rho):
    return (np.ones(rho.shape)-d_s_arr(rho))/rho - np.ones(rho.shape)

@njit
def Q_diff_vectorized(rho_1, rho_2):
    rho_avg = (rho_1 + rho_2) / 2
    alpha = np.pi / 2 - 1
    beta = alpha * (2 * alpha - 1) / (2 * alpha + 1)

    # Calculate dQ_drho in a vectorized way
    dQ_drho = ((alpha + 1) - (alpha + beta) * rho_avg + beta * rho_avg**2) / d_s_arr(rho_avg)
    return dQ_drho * (rho_1 - rho_2)

@njit
def compute_rho(f, delta_theta):
    return np.sum(f, axis=1) * delta_theta

@njit
def compute_p1(f, e1, delta_theta):
    return np.dot(f, e1) * delta_theta

@njit
def compute_U_x(f, rho_, p1_, delta_x, delta_theta, D_E, v_0, N_theta, d_s_, s_, Q_diff_vectorized):
    # Get the number of x values
    N_x = f.shape[0]
    
    # Create log_f array
    log_f = np.empty_like(f)
    for i in range(N_x):
        log_f[i] = np.log(f[i])
    
    # Compute diffusion term manually
    diffusion = np.empty((N_x - 1, f.shape[1]))
    for i in range(1, N_x):
        diffusion[i - 1] = (log_f[i] - log_f[i - 1]) / delta_x
    
    # Compute exclusion term
    exclusion = np.empty((N_x - 1, N_theta))
    q_diff = Q_diff_vectorized(rho_[1:], rho_[:-1]) / delta_x
    for j in range(N_theta):
        exclusion[:, j] = q_diff

    # Compute factor and average
    factor = (1 - rho_) / d_s_
    factor_avg = np.empty(N_x - 1)
    for i in range(N_x - 1):
        factor_avg[i] = 0.5 * (factor[i] + factor[i + 1])

    # Compute potential term
    #potential = np.empty((N_x - 1, N_theta))
    #for j in range(N_theta):
    #    potential[:, j] = factor_avg * V_diff

    # Compute p1_s_rho and average
    p1_s_rho = p1_ * s_ / d_s_
    p1_avg = np.empty(N_x - 1)
    for i in range(1, N_x):
        p1_avg[i - 1] = 0.5 * (p1_s_rho[i] + p1_s_rho[i - 1])

    # Compute activity
    activity = np.empty((N_x - 1, N_theta))
    cos_term = np.cos(np.arange(N_theta) * delta_theta)
    for j in range(N_theta):
        activity[:, j] = (p1_avg + cos_term[j])

    # Return the final computation
    return -D_E * (diffusion + exclusion) + v_0 * activity

@njit
def compute_U_theta(f, delta_theta, D_O):
    log_f = np.log(f)
    shifted_log_f = np.empty_like(log_f)
    shifted_log_f[:, :-1] = log_f[:, 1:]   
    shifted_log_f[:, -1] = log_f[:, 0]  
    angular_diffusion = (shifted_log_f - log_f) / delta_theta
    return -D_O * angular_diffusion[:-1, :]

@njit
def compute_F_x(f, U_x, d_s_rho, delta_x):
    # Create f_next by manually shifting f upwards along axis 0
    f_next = np.empty_like(f[:-1, :])
    for i in range(f.shape[0] - 1):
        f_next[i, :] = f[i + 1, :]

    # Manually compute the averaged d_s_rho values
    d_s_avg = np.empty((d_s_rho.shape[0] - 1, 1))
    for i in range(d_s_rho.shape[0] - 1):
        d_s_avg[i, 0] = 0.5 * (d_s_rho[i] + d_s_rho[i + 1])

    # Initialize the result array
    F_x = np.empty_like(f_next)

    # Apply upwind scheme conditionally
    for i in range(f_next.shape[0]):
        for j in range(f_next.shape[1]):
            if U_x[i, j] >= 0:
                F_x[i, j] = U_x[i, j] * f[i, j] * d_s_avg[i, 0]
            else:
                F_x[i, j] = U_x[i, j] * f_next[i, j] * d_s_avg[i, 0]

    return F_x

@njit
def compute_F_theta(f, U_theta):
    # Create f_shifted by manually shifting f left along axis 1
    f_shifted = np.empty_like(f)
    for i in range(f.shape[1] - 1):
        f_shifted[:, i] = f[:, i + 1]
    f_shifted[:, -1] = f[:, 0]  # Wrap around for last column to first

    # Calculate F_theta with the upwind scheme
    F_theta = np.empty_like(f)
    for i in range(f.shape[0] - 1):
        for j in range(f.shape[1]):
            F_theta[i, j] = U_theta[i, j] * (f[i, j] + f_shifted[i, j]) / 2

    return F_theta

@njit
def roll_2d(arr, shift, axis):
    """Roll a 2D array along the specified axis."""
    if axis == 0:  # Roll along the first axis (rows)
        return np.concatenate((arr[-shift:], arr[:-shift]), axis=0)
    elif axis == 1:  # Roll along the second axis (columns)
        return np.concatenate((arr[:, -shift:], arr[:, :-shift]), axis=1)
    else:
        raise ValueError("Axis must be 0 or 1.")

@njit
def compute_df_dt(F_x_, F_theta_, delta_x, delta_theta, N_x, d_s_):
    
    dF_x_dx = -(F_x_[1:N_x - 1, :] - F_x_[:N_x - 2, :]) / delta_x
    
    F_theta_rolled = roll_2d(F_theta_[1:N_x - 1, :], 1, axis=1)
    dF_theta_dtheta = -(F_theta_[1:N_x - 1, :] - F_theta_rolled) / delta_theta
    
    # Return the sum of derivatives
    return dF_x_dx + dF_theta_dtheta

class f:
    def __init__(self, N_x, N_theta, D_E=1.0, v_0=20.0, D_O=1.0, L_x=1.0) -> None:
        self.L_x = L_x
        self.N_x = N_x
        self.N_theta = N_theta
        self.delta_x = L_x / N_x
        self.delta_theta = 2 * np.pi / N_theta
        self.t = 0.0

        self.D_E = D_E
        self.v_0 = v_0
        self.D_O = D_O

        self.l_p = v_0 / D_O
        self.l_D = np.sqrt(D_E / D_O)
        self.Pe = self.l_p / self.l_D
        self.l = self.l_D / L_x

        self.f = np.zeros((N_x, N_theta))
        self.calc_time = 0.0
        self.e1 = np.cos(np.arange(self.N_theta) * self.delta_theta)

    #some methods for setting up the inital condition
    def set_random(self, phi, delta):
        self.f = np.copy(np.add((phi-delta/2)*np.ones((self.N_x, self.N_theta)), delta*np.random.rand(self.N_x, self.N_theta))*1/(2*np.pi))
    
    def set_sinusoidal(self):
        x = np.linspace(-np.pi, np.pi, self.N_x)  # Symmetric around 0
        sine_wave = np.sin(x*5)+2
        sin = np.tile(sine_wave, (self.N_theta, 1)).T
        self.f = np.copy(0.8*sin/(2*np.pi*3))

    def set_gaussian(self, std_dev=1.5):
        x = np.linspace(-2, 2, self.N_x)  # Symmetric around 0
        mean = 0
        gaussian_1d = np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
        gaussian_2d = np.tile(gaussian_1d, (self.N_theta, 1)).T
        self.f = 0.9*gaussian_2d/(2*np.pi)

    #methods to evolve the system (split up for ease of reading)
    def df_dt(self):
        rho_ = compute_rho(self.f, self.delta_theta)
        p1_ = compute_p1(self.f, self.e1, self.delta_theta)
        d_s_ = d_s_arr(rho_)
        s_ = s_arr(rho_)

        U_x_ = compute_U_x(self.f, rho_, p1_, self.delta_x, self.delta_theta, 
                           self.D_E, self.v_0, self.N_theta, d_s_, s_, Q_diff_vectorized)
        U_theta_ = compute_U_theta(self.f, self.delta_theta, self.D_O)

        F_x_ = compute_F_x(self.f, U_x_, d_s_, self.delta_x)
        F_theta_ = compute_F_theta(self.f, U_theta_)
        

        a_x = np.max(np.abs(U_x_))
        a_theta = np.max(np.abs(U_theta_))


        return a_x, a_theta, compute_df_dt(F_x_, F_theta_, self.delta_x, self.delta_theta, self.N_x, d_s_)

    def update(self):
        
        self.f[0, :] = self.f[-2, :]
        self.f[-1, :] = self.f[1, :]
        
        a_x, a_theta, df_dt_ = self.df_dt()
        a_x = max(a_x, 0.1)
        a_theta = max(a_theta, 0.1)

        delta_t = min(self.delta_x / (a_x * 6), self.delta_theta / (a_theta * 6), 10**(-4))
        
        self.f[1:-1, :] += df_dt_ * delta_t
        self.t += delta_t
        

        return delta_t

    def evolve(self, N_t):
        self.history_rho = []
        self.history_p1 = []
        self.history_direction = []

        for n in range(N_t):
            if n % 100 == 0:
                rho_ = compute_rho(self.f, self.delta_theta)
                self.history_rho.append(rho_)
                p1_ = compute_p1(self.f, self.e1, self.delta_theta)
                self.history_p1.append(p1_)
                self.history_direction.append(np.divide(p1_, rho_))
                time_since_last_record = 0.0  
            delta_t = self.update()
            self.t += delta_t

    def evolve_to_time(self, t, recording_steps):
        self.t=0.0
        self.history_rho = []
        self.history_p1 = []
        self.history_direction = []

        step = 0.0
        time_since_last_record = 0.0
        while self.t <= t:
            if time_since_last_record >= recording_steps:
                rho_ = compute_rho(self.f, self.delta_theta)
                self.history_rho.append(rho_)
                p1_ = compute_p1(self.f, self.e1, self.delta_theta)
                self.history_p1.append(p1_)
                self.history_direction.append(np.divide(p1_, rho_))
                time_since_last_record = 0.0       
            delta_t = self.update()
            self.t += delta_t
            time_since_last_record += delta_t
    
    #analysis of final results
    def plot_history(self, y_breaks, x_breaks):
        N_t = len(self.history_direction)

        x, y = np.meshgrid(np.linspace(0, self.N_x, int(self.N_x/x_breaks)),np.linspace(0, N_t, int(N_t/y_breaks))) 
        u = np.array(self.history_direction)[::y_breaks, 0::x_breaks]
        v = 0
        
        # Plotting Vector Field with QUIVER 
        plt.figure(figsize=(10,5))
        #plt.quiver(x, y, u, v, color='g') 
        plt.title(f't = {self.t}') 
        plt.grid() 
        #plt.imshow(p_history[:N_t], cmap='coolwarm', aspect='auto')#
        plt.imshow(self.history_rho[:N_t], cmap='coolwarm', aspect='auto', vmin=0.0, vmax=1.0)
        plt.colorbar()

    def rho_asymmetry(self, rho_):
        return np.sum(np.abs(rho_[1:-1] - np.flip(rho_[1:-1])))/np.sum(np.abs(rho_[1:-1] + np.flip(rho_[1:-1])))
    
    def phi(self):
        rho_ = rho_ = compute_rho(self.f, self.delta_theta)
        return np.sum(rho_[1:-1])*self.L_x/(self.N_x - 2)
   
