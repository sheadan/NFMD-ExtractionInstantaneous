# Model Examples
import warnings
from scipy import optimize
warnings.filterwarnings('ignore')

# Setting parameters for models with known constants
tp = 0.001638
decay_lambda = 1080.80


# 4 DoF Model
model5 = lambda p, t: np.heaviside(t-p[2],0.5)*p[0]*np.exp(-p[3]*(t-p[2]))*np.expm1(-(t-p[2])/p[1]) # Target function
p0_5 = [1, 5e-6, 0.001636, 1080] # Initial guess for the parameters

# 3 DoF (lambda=0)
model4 = lambda p, t: np.heaviside(t-tp,0.5)*p[0]*np.expm1(-(t-tp)/p[1]) # Target function
p0_4 = [1, 5e-6, 0.001636] # Initial guess for the parameters

# 2 DoF (fixed t_perturb and lambda decay)
model3 = lambda p, t: np.heaviside(t-tp,0.5)*p[0]*np.exp(-decay_lambda*(t-tp))*np.expm1(-(t-tp)/p[1]) # Target function
p0_3 = [1, 5e-6] # Initial guess for the parameters

# 3 DoF (fixed decay constant)
model2 = lambda p, t: np.heaviside(t-p[2],0.5)*p[0]*np.exp(-decay_lambda*(t-p[2]))*np.expm1(-(t-p[2])/p[1]) # Target function
p0_2 = [1, 5e-6, 0.001636] # Initial guess for the parameters

# 3 DoF (fixed tp)
model1 = lambda p, t: np.heaviside(t-tp,0.5)*p[0]*np.exp(-p[2]*(t-tp))*np.expm1(-(t-tp)/p[1]) # Target function
p0_1 = [1, 1e-6, 1080] # Initial guess for the parameters

# Error function to minimize
errfunc = lambda p, x, y, model: model(p, x) - y # Distance to the target function

# An example of fitting the model to the data, where 'mean' is a vector with the data:
# the returned 'p' variable holds the fitted parameters, while 'success' is an integer indicating fit success
p1, success = optimize.leastsq(errfunc, p0_1[:], args=(nfmd_ts, mean, model1))
p2, success = optimize.leastsq(errfunc, p0_2[:], args=(nfmd_ts, mean, model2))
p3, success = optimize.leastsq(errfunc, p0_3[:], args=(nfmd_ts, mean, model3))
p4, success = optimize.leastsq(errfunc, p0_4[:], args=(nfmd_ts, mean, model4))
p5, success = optimize.leastsq(errfunc, p0_5[:], args=(nfmd_ts, mean, model5))