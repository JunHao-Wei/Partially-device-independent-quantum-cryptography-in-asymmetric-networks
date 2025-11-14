import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy import Symbol, Mul, Pow, simplify
import time
from math import sqrt, log2, log, pi, cos, sin
import ncpol2sdpa as ncp
from sympy.physics.quantum.dagger import Dagger
import mosek
import chaospy
import picos
import cvxopt as cvx
from ncpol2sdpa.picos_utils import convert_to_picos
from collections import OrderedDict


start_time = time.time()

def h(p):
	"""
	Returns the binary entropy function h(p)
	"""
	if p==0 or p==1:
		return 0
	return -p*log2(p) - (1-p)*log2(1-p)
	
def cond_ent(joint, marg):
    """
    Returns H(A|B) = H(AB) - H(B)

    Inputs:
        joint    --     joint distribution on AB
        marg     --     marginal distribution on B
    """

    hab, hb = 0.0, 0.0

    for prob in joint:
        if 0.0 < prob < 1.0:
            hab += -prob*log2(prob)

    for prob in marg:
        if 0.0 < prob < 1.0:
            hb += -prob*log2(prob)

    return hab - hb
def HAgB(V):
    """
    Computes the error correction term in the key rate for a given visibility
    """
    q00 = (1+V)/4
    q01 = (1-V)/4
    q10 = (1-V)/4
    q11 = (1+V)/4

    qb0 = 1/2
    qb1 = 1/2

    qjoint = [q00,q01,q10,q11]
    qmarg = [qb0,qb1]

    return cond_ent(qjoint, qmarg)
    
def get_subs():
    """
    Returns any substitution rules to use with ncpol2sdpa. E.g. projections and
    commutation relations.
    """
    subs = {}
    # Get Bob's projective measurement constraints
    subs.update(ncp.projective_measurement_constraints(B))

    # Note that Bob's operators should All commute with Eve's ops
    for b in ncp.flatten(B):
        for z in Z:
            subs.update({z*b : b*z, Dagger(z)*b : b*Dagger(z)})

    return subs

def get_extra_monomials():
    """
    Returns additional monomials to add to sdp relaxation.
    """

    monos = []
    # Add BZ
    # ~ ZZ = Z + [Dagger(z) for z in Z]
    # ~ Bflat = ncp.flatten(B)
    # ~ for b in Bflat:
        # ~ for z in ZZ:
            # ~ monos += [b*z]

    # Add monos appearing in objective function
    # ~ for z in Z:
        # ~ monos += [Dagger(z)*z]

    return monos[:]

def generate_quadrature(m):
    """
    Generates the Gaussian quadrature nodes t and weights w. Due to the way the
    package works it generates 2*M nodes and weights. Maybe consider finding a
    better package if want to compute for odd values of M.

        m    --    number of nodes in quadrature / 2
    """
    t, w = chaospy.quadrature.radau(m, chaospy.Uniform(0, 1), 1)
    t = t[0]
    return t, w

# Obtain the dictionary of monomials from the file "monomials.txt" that stores the monomials
def load_monomial_dict(filename):
    monomial_dict = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            parts = line.split(maxsplit=1)
            if len(parts) == 1:
                idx, monomial = int(parts[0]), ""
            else:
                idx, monomial = int(parts[0]), parts[1]
            monomial_dict[monomial] = idx
    return monomial_dict

# Helper function to get the hermitian_conjugate of a monomial
def hermitian_conjugate(term: str) -> str:
    if '^' in term:
        base, power = term.split('^')
        if base.startswith('Z'):
            if base.endswith('T'):
                base = base[:-1]
            else:
                base = base + 'T'
            return f"{base}^{power}"
        else:
            return term 

    if term.startswith('Z'):
        if term.endswith('T'):
            return term[:-1]
        else:
            return term + 'T'
    return term

# Genrate the moment matrix Gamma and determine the hermicity of blocks
def Generate_moment_matrix(monomial_list, HermitianVars, ComplexVars):
    # Define the Hermitian conjugated monomial list
    hc_monomial_list = []
    for m in monomial_list:
        terms = m.split('*')
        reversed_terms = terms[::-1]
        conjugated_terms = [hermitian_conjugate(term) for term in reversed_terms]
        conjugated_m = '*'.join(conjugated_terms)
        hc_monomial_list.append(conjugated_m)

    B0, B1, Z0, Z1, Z0T, Z1T = sp.symbols("B0 B1 Z0 Z1 Z0T Z1T", commutative=False)
    local_dict = {
        'B0': B0, 'B1': B1,
        'Z0': Z0, 'Z1': Z1,
        'Z0T': Z0T, 'Z1T': Z1T
    }

    N = len(monomial_list)
    matrix = [[None for _ in range(N)] for _ in range(N)]

    for i, hc_mono_str in enumerate(hc_monomial_list):
        for j, mono_str in enumerate(monomial_list):
            full_str = hc_mono_str + "*" + mono_str
            full_str = full_str.replace("^", "**")
            try:
                expr = parse_expr(full_str, local_dict=local_dict, evaluate=False)
            except:
                expr = full_str
    # Apply commutative and projective substituion rules
            for B in [B0, B1]:
                for power in [2,3,4]:
                    expr = expr.replace(Pow(B, power), B)
            if isinstance(expr, Mul):
                b_terms = []
                z_terms = []
                for arg in expr.args:
                    if arg in [B0, B1]:
                        b_terms.append(arg)
                    else:
                        z_terms.append(arg)
                expr = Mul(*b_terms) * Mul(*z_terms)

            for B in [B0, B1]:
                for power in [2,3,4]:
                    expr = expr.replace(Pow(B, power), B)
            expr_str = str(expr).replace("**", "^")
            matrix[i][j] = expr_str

    def convert_key_to_string(key):
        return key.replace("^", "p").replace("*", "_")
    
    # Check whether the matrix element is a monomial or hermitian conjugated monomial. Normalize the matrix elements to be monomials.
    replaced_matrix = [["" for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            entry = matrix[i][j]

            if entry in monomial_dict:
                replaced_matrix[i][j] = convert_key_to_string(entry)
                continue
            matched = False
            for key in monomial_dict:
                terms = key.split('*')
                reversed_terms = terms[::-1]
                conjugated_terms = [hermitian_conjugate(term) for term in reversed_terms]
                conjugated_key = '*'.join(conjugated_terms)
                terms_p = conjugated_key.split("*")
                b_terms = [t for t in terms_p if t in ("B0", "B1")]
                other_terms = [t for t in terms_p if t not in ("B0", "B1")]

                if entry == "*".join(b_terms + other_terms):
                    replaced_matrix[i][j] = convert_key_to_string(key) + ".H"
                    matched = True
                    break

            if not matched:
                if i == 0 and j == 0:
                    replaced_matrix[i][j] = 'rhoA'
                else:
                    print(f"[Check] Unmatched entry at ({i},{j}): {entry}")
                    replaced_matrix[i][j] = entry
    # ~ Check Hermiicity        
    # ~ print(replaced_matrix)
    # ~ for i in range(N):
        # ~ for j in range(N):
            # ~ print(replaced_matrix[i][j], replaced_matrix[j][i])
    # ~ return replaced_matrix    
    
    rows = []

    for i, row in enumerate(replaced_matrix):
        picos_row = None
        for j, entry in enumerate(row):
            is_conj = entry.endswith(".H")
            key = entry[:-2] if is_conj else entry
            if key in HermitianVars:
                var = HermitianVars[key]
            elif key in ComplexVars:
                var = ComplexVars[key]
            else:
                raise KeyError(f"Variable '{key}' not found in either dictionary at ({i},{j})")
            if is_conj:
                var = var.H

            picos_row = var if picos_row is None else (picos_row & var)

        rows.append(picos_row)

    picos_matrix = rows[0]
    for row in rows[1:]:
        picos_matrix = picos_matrix // row
    return picos_matrix


def Generate_variables(monomial_dict, dimA=2):	
	Hermitian_M = []
	Complex_M = []
	
	# Check the hermicity of each monomial
	for key in monomial_dict:
	    if key == '':
	        Hermitian_M.append(key)
	        continue
	    terms = key.split('*')
	    reversed_terms = terms[::-1]
	    conjugated_terms = [hermitian_conjugate(term) for term in reversed_terms]
	    conjugated_key = '*'.join(conjugated_terms)
	    if conjugated_key == key:
	        Hermitian_M.append(key)
	    else:
	        Complex_M.append(key)
	# ~ print("Hermitian_M =", Hermitian_M)
	# ~ print("Complex_M =", Complex_M)
	    
	HermitianVars = {} # Dictionary that used to store the Hermitian block matrix variables for picos optimization
	ComplexVars = {} # Dictionary that used to store the non-Hermitian block matrix variables for picos optimization

	def make_variable_name(expr: str) -> str:
	    expr = expr.replace('*', '_') # replace the '*' in variable names by '_'
	    expr = expr.replace('^', 'p')  # replace the '^' in variable names by 'p'
	    return expr if expr else 'rhoA'  # '' is replaced by 'rhoA'
	
	for expr in Hermitian_M:
	    var_name = make_variable_name(expr)
	    HermitianVars[var_name] = picos.HermitianVariable(var_name, dimA)
	
	for expr in Complex_M:
	    var_name = make_variable_name(expr)
	    ComplexVars[var_name] = picos.ComplexVariable(var_name, (dimA, dimA))
	return HermitianVars, ComplexVars


# Use ncpol2sdpa to generate the monomials and the moment matrix for further processing
LEVEL = 1                        # NPA relaxation level
VERBOSE = 2
B_config = [2,2]
B = [Bj for Bj in ncp.generate_measurements(B_config, 'B')]
Z = ncp.generate_operators('Z', 2, hermitian=0)

M = 1                            # Number of nodes / 2 in gaussian quadrature
T, W = generate_quadrature(M)

substitutions = {}            # substitutions to be made (e.g. projections)
moment_ineqs = []            # Moment inequalities (e.g. Tr[rho CHSH] >= c)
moment_eqs = []                # Moment equalities (not needed here)
op_eqs = []                    # Operator equalities (not needed here)
op_ineqs = []                # Operator inequalities (e.g. Id - A00 >= 0 -- we don't include for speed)
extra_monos = []            # Extra monomials to add to the relaxation beyond the level.
# Get the relevant substitutions
substitutions = get_subs()
# Get any extra monomials we wanted to add to the problem
extra_monos = get_extra_monomials()
ops = ncp.flatten([B,Z])
sdp = ncp.SdpRelaxation(ops, verbose = VERBOSE-1, normalized=True, parallel=0)
sdp.get_relaxation(level = LEVEL,
    equalities = op_eqs[:],
    inequalities = op_ineqs[:],
    momentequalities = moment_eqs[:],
    momentinequalities = moment_ineqs[:],
    substitutions = substitutions,
    extramonomials = extra_monos)

monomial_list = ncp.get_monomials(ops, LEVEL)+get_extra_monomials()
monomial_list = list(OrderedDict.fromkeys(
    ncp.nc_utils.convert_monomial_to_string(m) for m in monomial_list
    if m not in get_subs()
))
# ~ sdp.write_to_file("1sDIQKD_BB84meas_no-postselection.csv") # save the moment matrix for reference 
sdp.save_monomial_index("1sDIQKD_BB84meas_monomials.txt")


# Define the picos optimization problem	
monomial_dict = load_monomial_dict("1sDIQKD_BB84meas_monomials.txt")
# ~ print(monomial_dict)

HermitianVars, ComplexVars = Generate_variables(monomial_dict, dimA=2)
# ~ print(HermitianVars, len(HermitianVars))
# ~ print(ComplexVars, len(ComplexVars))

picos_matrix = Generate_moment_matrix(monomial_list, HermitianVars, ComplexVars) 

# Add constraints
I2 = picos.Constant([[1,0],[0,1]])
A0 = picos.Constant([[1,0],[0,0]])
A1 = picos.Constant([[1/2,1/2],[1/2,1/2]])

V_list = np.linspace(0.75, 1 - 1e-4, 15).tolist()
P_dict = {}
results = []
BB84_results = []
for idx, V in enumerate(V_list):
    P = picos.Problem(name=f"P_{round(V, 4)}")
    P.add_constraint((picos_matrix + picos_matrix.H) / 2 >> 0)
    Score_cons = P.add_list_of_constraints([picos.trace(A1 * HermitianVars['B1']) == (1+V)/4, \
                               picos.trace((I2-A1) * HermitianVars['B1']) == (1-V)/4, \
                               picos.trace(A1 * (HermitianVars['rhoA'] - HermitianVars['B1'])) == (1-V)/4, \
                               picos.trace((I2-A1) * (HermitianVars['rhoA'] - HermitianVars['B1'])) == (1+V)/4
    ])

	# Set the objective and solve the SDP
    Entr = 0
    for i in range(2*M-1):
        P.set_objective("min", picos.trace(A0 * (ComplexVars['Z0'] + ComplexVars['Z0'].H + (1-T[i]) * HermitianVars['Z0T_Z0']) + T[i] * HermitianVars['Z0_Z0T']) + \
                               picos.trace((I2-A0) * (ComplexVars['Z1'] + ComplexVars['Z1'].H + (1-T[i]) * HermitianVars['Z1T_Z1']) + T[i] * HermitianVars['Z1_Z1T']))
        P.solve(solver = "mosek")
        Entr += (1+float(P)) * W[i] / (T[i] * log(2))

    Key_rate = Entr - HAgB(V)
    results.append(Key_rate)
    BB84_results.append(1 - 2*h((1-V)/2))
    print(V, Key_rate, 1 - 2*h((1-V)/2))
    P_dict[f"P_{round(V, 4)}"] = P
# ~ print(results)

import matplotlib.pyplot as plt
QBER = [(1 - V) / 2 for V in V_list]
plt.figure(figsize=(8,5))

plt.plot(QBER, results, label='1sDI', marker='o')
plt.plot(QBER, BB84_results, label='BB84', marker='s')
plt.ylim(0.0, 1.0)
plt.xlim(0, 0.125)
plt.xlabel('QBER')
plt.ylabel('key rate')
plt.title('Key Rate vs QBER')
plt.legend()
plt.grid(False)
plt.minorticks_on()
plt.tick_params(which='both', direction='in', top=True, right=True)
plt.tick_params(which='major', length=6)
plt.tick_params(which='minor', length=3, color='gray')
end_time = time.time()
print("Time cost:", end_time - start_time)
plt.show()
