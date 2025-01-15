import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sympy import symbols, lambdify

p = 0.05
T40 = 1
T48 = 0.94


#@title Create sympy symbols

# SRC pair Numbers
r_np48_pp48 = sp.symbols('r_np48_pp48')  # np48/pp48
r_nn48_pp48 = sp.symbols('r_nn48_pp48')  # nn48/pp48
r_np40_pp40 = sp.symbols('r_np40_pp40')  # np40/pp40
r_nn40_pp40 = sp.symbols('r_nn40_pp40')  # nn40/pp40
r_pp48_pp40 = sp.symbols('r_pp48_pp40')  # pp48/pp40

# Cross Sections
sigma_ratio = sp.symbols('sigma')  # sigma_p / sigma_n


# Transparency terms
T_p_48, T_p_40 = sp.symbols('T_p_48 T_p_40')  # T_p terms
T_n_48, T_n_40 = sp.symbols('T_n_48 T_n_40')  # T_n terms
T_np_48, T_np_40 = sp.symbols('T_np_48 T_np_40')  # T_np terms
T_pp_48, T_pp_40 = sp.symbols('T_pp_48 T_pp_40')  # T_pp terms
P = sp.symbols('P')  # Single charge exchange

#@title Defining rations (R values) from approximate expressions

R_ee_p = ((r_np48_pp48*r_pp48_pp40 + 2*r_pp48_pp40)*sigma_ratio*T_p_48 +
          (r_np48_pp48*r_pp48_pp40 + 2*r_nn48_pp48*r_pp48_pp40)*T_p_48*P) / \
         ((r_np40_pp40 + 2)*sigma_ratio*T_p_40 +
          (r_np40_pp40 + 2*r_nn40_pp40)*T_p_40*P)

R_ee_n = ((r_np48_pp48*r_pp48_pp40 + 2*r_nn48_pp48*r_pp48_pp40)*T_n_48 +
          (r_np48_pp48*r_pp48_pp40 + 2*r_pp48_pp40)*sigma_ratio*T_n_48*P) / \
         ((r_np40_pp40 + 2*r_nn40_pp40)*T_n_40 +
          (r_np40_pp40 + 2)*sigma_ratio*T_n_40*P)

R_ee = (r_np48_pp48*r_pp48_pp40*(1 + sigma_ratio) +
        2*r_pp48_pp40*sigma_ratio +
        2*r_nn48_pp48*r_pp48_pp40) / \
       (r_np40_pp40*(1 + sigma_ratio) +
        2*sigma_ratio +
        2*r_nn40_pp40)

R_ee_np = ((r_np48_pp48*r_pp48_pp40 + 2*r_nn48_pp48*r_pp48_pp40*P)*T_np_48 +
           2*r_pp48_pp40*P*sigma_ratio*T_np_48) / \
          ((r_np40_pp40 + 2*r_nn40_pp40*P)*T_np_40 +
           2*P*sigma_ratio*T_np_40)

R_ee_pp = (2*r_pp48_pp40*sigma_ratio*T_pp_48 +
           r_np48_pp48*r_pp48_pp40*P*T_pp_48) / \
          (2*sigma_ratio*T_pp_40 +
           r_np40_pp40*P*T_pp_40)

R1= (2*r_pp48_pp40*sigma_ratio*T_pp_48 +
           r_np48_pp48*r_pp48_pp40*P*T_pp_48)
R2= (2*sigma_ratio*T_pp_40 +
           r_np40_pp40*P*T_pp_40)

R_p_48 = (2*sigma_ratio*T_pp_48 +
          r_np48_pp48*P*T_pp_48) / \
         ((r_np48_pp48 + 2)*sigma_ratio*T_p_48 +
          (r_np48_pp48 + 2*r_nn48_pp48)*T_p_48*P)

R_n_48 = ((r_np48_pp48 + 2*r_nn48_pp48*P)*T_np_48 +
          2*P*sigma_ratio*T_np_48) / \
         ((r_np48_pp48 + 2*r_nn48_pp48)*T_n_48 +
          (r_np48_pp48 + 2)*sigma_ratio*T_n_48*P)

R_p_40 = (2*sigma_ratio*T_pp_40 +
          r_np40_pp40*P*T_pp_40) / \
         ((r_np40_pp40 + 2)*sigma_ratio*T_p_40 +
          (r_np40_pp40 + 2*r_nn40_pp40)*T_p_40*P)

R_n_40 = ((r_np40_pp40 + 2*r_nn40_pp40*P)*T_np_40 +
          2*P*sigma_ratio*T_np_40) / \
         ((r_np40_pp40 + 2*r_nn40_pp40)*T_n_40 +
          (r_np40_pp40 + 2)*sigma_ratio*T_n_40*P)


# List of ratio expressions
ratio_exps = [R_ee_p, R_ee_n, R_ee, R_ee_np, R_ee_pp, R_p_48, R_n_48, R_p_40, R_n_40]

#@title Substituting to find ratio values


# Get ratios (R values) from constants + pair numbers
def get_ratios(sub):
  ratio_vals = [ratio_exp.subs(sub) for ratio_exp in ratio_exps]
  return ratio_vals

# Get ratios (R values) from pair numbers
def get_ratios_from_pair_nums(PN):
  np48, nn48, pp48, np40, nn40, pp40 = tuple(PN)
  substitute = {r_np48_pp48: np48/pp48, r_nn48_pp48: nn48/pp48, r_np40_pp40: np40/pp40,
                      r_nn40_pp40: nn40/pp40, sigma_ratio: 3,
                      r_pp48_pp40: pp48/pp40,
                      T_p_48: T48, T_p_40: T40,
                      T_n_48: T48, T_n_40: T40,
                      T_pp_48: T48, T_pp_40: T40,
                      T_np_48: T48, T_np_40: T40,
                      P: p}
  ratio_vals = [ratio_exp.subs(substitute) for ratio_exp in ratio_exps]
  return ratio_vals

def print_ratios(ratio_vals):
  print(f"Reep: {ratio_vals[0]}\n\
  Reen: {ratio_vals[1]}\n\
  Ree: {ratio_vals[2]}\n\
  Reenp: {ratio_vals[3]}\n\
  Reepp: {ratio_vals[4]}\n\
  Rp48: {ratio_vals[5]}\n\
  Rn48: {ratio_vals[6]}\n\
  Rp40: {ratio_vals[7]}\n\
  Rn40: {ratio_vals[8]}")

def get_ratio_print_msg(ratio_vals):
  return f"Reep: {ratio_vals[0]}\n\
Reen: {ratio_vals[1]}\n\
Ree: {ratio_vals[2]}\n\
Reenp: {ratio_vals[3]}\n\
Reepp: {ratio_vals[4]}\n\
Rp48: {ratio_vals[5]}\n\
Rn48: {ratio_vals[6]}\n\
Rp40: {ratio_vals[7]}\n\
Rn40: {ratio_vals[8]}"


#@title Recreating symbols (TODO: remove)

np_48, nn_48, pp_48, np_40, nn_40, pp_40 = symbols('#np_48 #nn_48 #pp_48 #np_40 #nn_40 #pp_40')

sigma_p, sigma_n = symbols('sigma_p sigma_n')
Tp_40, Tp_48 = symbols('T_p_40 T_p_48')
Tn_40, Tn_48 = symbols('T_n_40 T_n_48')
Tnp_40, Tnp_48 = symbols('T_np_40 T_np_48')
Tpp_40, Tpp_48 = symbols('T_pp_40 T_pp_48')
P = symbols('P')

Reep, Reen, Ree, Reenp, Reepp, Rp48, Rn48, Rp40, Rn40 = symbols('R_ee\'p R_ee\'n R_ee R_ee\'np R_ee\'pp R_p_48 R_n_48 R_p_40 R_n_40')


#@title Creating matrix of coefficients


# Equations in symbolic form

eqs = [

    Reep * ((np_40 + 2 * nn_40) * sigma_p * Tp_40 + (np_40 + 2 * nn_40) * sigma_n * Tp_40 * P) -
    ((np_48 + 2 * pp_48) * sigma_p * Tp_48 + (np_48 + 2 * nn_48) * sigma_n * Tp_48 * P),

    Reen * ((np_40 + 2 * nn_40) * sigma_n * Tn_40 + (np_40 + 2 * nn_40) * sigma_p * Tn_40 * P) -
    ((np_48 + 2 * nn_48) * sigma_n * Tn_48 + (np_48 + 2 * pp_48) * sigma_p * Tn_48 * P),

    Ree * (np_40 * (sigma_n + sigma_p) + 2 * nn_40 * sigma_p + 2 * nn_40 * sigma_n) -
    (np_48 * (sigma_n + sigma_p) + 2 * pp_48 * sigma_p + 2 * nn_48 * sigma_n),

    Reenp * ((np_40 + 2 * nn_40 * P) * sigma_n * Tnp_40 + 2 * nn_40 * P * sigma_p * Tnp_40) -
    ((np_48 + 2 * nn_48 * P) * sigma_n * Tnp_48 + 2 * pp_48 * P * sigma_p * Tnp_48),

    Reepp * (2 * nn_40 * sigma_p * Tpp_40 + np_40 * sigma_n * P * Tpp_40) -
    (2* pp_48 * sigma_p * Tpp_48 + np_48 * sigma_n * P * Tpp_48),

    Rp48 * ((np_48 + 2 * pp_48) * sigma_p * Tp_48 + (np_48 + 2 * nn_48) * sigma_n * Tp_48 * P) -
    (2 * pp_48 * sigma_p * Tpp_48 + np_48 * sigma_n * P * Tpp_48),

    Rn48 * ((np_48 + 2 * nn_48) * sigma_n * Tn_48 + (np_48 + 2 * pp_48) * sigma_p * Tn_48 * P) -
    ((np_48 + 2 * nn_48 * P) * sigma_n * Tnp_48 + 2 * pp_48 * P * sigma_p * Tnp_48),

    Rp40 * ((np_40 + 2 * nn_40) * sigma_p * Tp_40 + (np_40 + 2 * nn_40) * sigma_n * Tp_40 * P) -
    (2 * nn_40 * sigma_p * Tpp_40 + np_40 * sigma_n * P * Tpp_40),

    Rn40 * ((np_40 + 2 * nn_40) * sigma_n * Tn_40 + (np_40 + 2 * nn_40) * sigma_p * Tn_40 * P) -
    ((np_40 + 2 * nn_40 * P) * sigma_n * Tnp_40 + 2 * nn_40 * P * sigma_p * Tnp_40),
]


variables = [np_48, nn_48, pp_48, np_40, nn_40]


coeffs = sp.Matrix([[sp.expand(eq).coeff(var) for var in variables] for eq in eqs])


# Extract a sub 5x5 matrix
A = 0
used_eqs = 0
# The 5 expressions to use for finding pair numbers
def select_ratio_eqs(R_eqs=[0,1,2,3,4]):
  global A, used_eqs
  used_eqs = R_eqs
  print(f"{len(R_eqs)} ratios were sent.")
  try:
    A = coeffs[R_eqs, :]
  except:
    print("Mismatch in number of ratios used.")
  return A

select_ratio_eqs()


#@title Substitute ratios and solve for pair numbers

# Substitute R values and solve
def find_pair_nums(ratio_list, norm_by_atom=False):
  if len(ratio_list) < 9:
    print("Not enough ratios")
    return
  vals = {
    Tp_48: T48, Tp_40: T40,
    Tn_48: T48, Tn_40: T40,
    Tnp_48: T48, Tnp_40: T40,
    Tpp_48: T48, Tpp_40: T40,
    sigma_p: 3, sigma_n: 1, P: p,
    Reep: ratio_list[0], Reen: ratio_list[1], Ree: ratio_list[2], Reenp: ratio_list[3], Reepp: ratio_list[4], Rp48: ratio_list[5], Rn48: ratio_list[6], Rp40: ratio_list[7], Rn40: ratio_list[8]
  }
  A_sub = A.subs(vals)
  U,S,VT = np.linalg.svd(np.array(A_sub, dtype=float))
  pair_nums = VT[-1]
  pair_nums = np.append(pair_nums, pair_nums[-1])
  src48 = np.sum(pair_nums[:3])
  src40 = np.sum(pair_nums[3:])
  if norm_by_atom:
    src = [src48,src48,src48,src40,src40,src40]
    for n in range(6):
      pair_nums[n] /= src[n]
  else: 
    pair_nums /= src40 + src48
  return (pair_nums, src48/src40, A_sub)


def print_pair_nums(pair_nums):
  print(f"np48: {pair_nums[0]}\n\
nn48: {pair_nums[1]}\n\
pp48: {pair_nums[2]}\n\
np40: {pair_nums[3]}\n\
nn40: {pair_nums[4]}\n\
pp40: {pair_nums[5]}")

def get_pair_nums_msg(pair_nums):
  return f"np48: {pair_nums[0]}\n\
nn48: {pair_nums[1]}\n\
pp48: {pair_nums[2]}\n\
np40: {pair_nums[3]}\n\
nn40: {pair_nums[4]}\n\
pp40: {pair_nums[5]}"

select_ratio_eqs([0,1,2,3,4])

# -------------------

# Define symbolic variables
x1, x2, x3, x4 = symbols('x1 x2 x3 x4')
sigma_p, sigma_n = symbols('sigma_p sigma_n')
Tp_48, Tp_40 = symbols('T_p_48 T_p_40')
Tn_48, Tn_40 = symbols('T_n_48 T_n_40')
Tnp_48, Tnp_40 = symbols('T_np_48 T_np_40')
Tpp_48, Tpp_40 = symbols('T_pp_48 T_pp_40')
P = symbols('P')

# Define equations
num1 = (x1 + 2 * x3) * sigma_p * Tp_48 + (x1 + 2 * x2) * sigma_n * Tp_48 * P
denom1 = (1 + 2 * x4) * sigma_p * Tp_40 + (1 + 2 * x4) * sigma_n * Tp_40 * P
res1 = num1 / denom1 - Reep

num2 = (x1 + 2 * x2) * sigma_n * Tn_48 + (x1 + 2 * x3) * sigma_p * Tn_48 * P
denom2 = (1 + 2 * x4) * sigma_n * Tn_40 + (1 + 2 * x4) * sigma_p * Tn_40 * P
res2 = num2 / denom2 - Reen

num3 = x1 * (sigma_n + sigma_p) + 2 * x3 * sigma_p + 2 * x2 * sigma_n
denom3 = (sigma_n + sigma_p) + 2 * x4 * sigma_p + 2 * x4 * sigma_n
res3 = num3 / denom3 - Ree

num4 = (x1 + 2 * x2 * P) * sigma_n * Tnp_48 + 2 * x3 * P * sigma_p * Tnp_48
denom4 = (1 + 2 * x4 * P) * sigma_n * Tnp_40 + 2 * x4 * P * sigma_p * Tnp_40
res4 = num4 / denom4 - Reenp

num5 = 2 * x3 * sigma_p * Tpp_48 + x1 * sigma_n * P * Tpp_48
denom5 = 2 * x4 * sigma_p * Tpp_40 + x4 * sigma_n * P * Tpp_40
res5 = num5 / denom5 - Reepp

# Create residual functions
params = [x1, x2, x3, x4] # variables
constants_sym = [sigma_p, sigma_n, Tp_48, Tp_40, Tn_48, Tn_40, Tnp_48, Tnp_40, Tpp_48, Tpp_40, P, Reep, Reen, Ree, Reenp, Reepp]

residuals_sym = [res1, res2, res3, res4, res5]
residuals_funcs = [lambdify(params + constants_sym, res) for res in residuals_sym] # minimize eq error

# Residual function for least_squares
def residuals(vars, *args):
    x1, x2, x3, x4 = vars
    constants = args
    res_funcs = [residuals_funcs[i] for i in range(len(residuals_funcs)) if i in used_eqs]
    return [func(x1, x2, x3, x4, *constants) for func in res_funcs]

# Example usage: Substitute known constants
known_constants = [3, T40, T48, T40, T48, T40, T48, T40, T48, T40, p]

def find_pair_nums_r(ratios, norm_by_atom=False):
  # Solve using least_squares
  initial_guess = [1, 2/7, 1/7, 1/14]
  values = known_constants.copy()
  ratio_list = [0]*5 # 5 is curr max # of ratios usable here
  for i in range(len(ratio_list)):
    if i in used_eqs:
      ratio_list[i] = ratios[i]
  values += ratio_list
  print("vals", values)
  result = least_squares(residuals, initial_guess, bounds=(0,np.inf), args=tuple(values), ftol=0.0001)

  x1_sol, x2_sol, x3_sol, x4_sol = result.x
  
  pair_nums = np.array([x1_sol, x2_sol, x3_sol, 1, x4_sol, x4_sol])
  src48 = np.sum(pair_nums[:3])
  src40 = np.sum(pair_nums[3:])
  if norm_by_atom:
    src = [src48,src48,src48,src40,src40,src40]
    for n in range(6):
      pair_nums[n] /= src[n]
  return (pair_nums, src48/src40)

