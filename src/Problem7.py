import numpy as np

mu = 3
sigma = 4
alpha = 2
beta = 5
w1 = 0.6
w2 = 0.4
m = 100



da = np.random.gumbel(alpha, beta, (m, 1))
db =  np.random.normal(mu, sigma, (m,1))
dx = w1 * da + (1-w1) * db;
# since that the w1 + w2 = 1



# EM Algorighm for (a)
step = 0
max_step = 10**4

mu = 4
alpha = 3
sigma = 5
beta = 3
w1 = 0.8
lr = 10**(-7)

t_mu = np.zeros((max_step, 1))
t_sigma = np.zeros((max_step, 1))
t_alpha = np.zeros((max_step, 1))
t_beta = np.zeros((max_step, 1))
t_w1 = np.zeros((max_step, 1))


# functions

def gussian(x, mu, sigma):
    p_x = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((x - mu) ** 2) / (2 * (sigma ** 2)))
    return p_x

def gumbel(x, alpha, beta):
    p_x = 1 / beta * np.exp(-(x - alpha)/beta)* np.exp(-np.exp(-(x - alpha)/beta))
    return p_x


while (step <= max_step - 1):
    p_x_1 = gumbel(dx, alpha, beta)
    p_x_2 = gussian(dx, mu, sigma)
    p_x_w = w1 * p_x_1 + (1 - w1) * p_x_2
    p_1_x = w1 * p_x_1 / p_x_w
    p_2_x = (1 - w1) * p_x_2 / p_x_w
    
    w1_new = sum(p_2_x) / m
    t_w1[step, 0] = 1 - w1_new
    
    
  
    alpha_new = alpha - lr * sum((1/beta) - np.exp(-(dx-alpha)/beta)*1/beta);
    t_alpha[step, 0]= alpha_new
    
    
    beta_new = beta - lr * ((-(m/beta) + sum((dx-alpha)/(beta**2)))+ sum((np.exp(-(dx-alpha)/beta))*(dx-alpha)/(beta**2)))
    t_beta[step, 0] = beta_new
    
    mu_tmp = (p_2_x * dx)
    mu_new = sum(mu_tmp) / sum(p_2_x)
    t_mu[step, 0] = mu_new
    
    sigma_tmp = (p_2_x * ((dx - mu) ** 2))
    sigma_new = np.sqrt(sum(sigma_tmp) / sum(p_2_x))
    t_sigma[step, 0] = sigma_new
    step = step + 1

    w1 = w1_new
    alpha = alpha_new
    beta = beta_new
    mu = mu_new
    sigma = sigma_new



print('w1 : ', w1)
print('alpha:', alpha)
print('beta:', beta)
print('mu:', mu)
print('sigma:',sigma)
print('\n')

print('mean w1:', np.mean(t_w1))
print('mean alpha:', np.mean(t_alpha))
print('mean beta:', np.mean(t_beta))
print('mean mu:', np.mean(t_mu))
print('mean sigma:', np.mean(t_sigma))
print('\n')

print('variance w1:', np.std(t_w1) ** 2)
print('variance alpha:', np.std(t_alpha) ** 2)
print('variance beta:', np.std(t_beta) ** 2)
print('variance mu:', np.std(t_mu) ** 2)
print('variance sigma:', np.std(t_sigma) ** 2)
print('\n')
