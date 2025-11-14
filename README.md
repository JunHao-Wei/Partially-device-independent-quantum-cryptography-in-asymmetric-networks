# Partially-device-independent-quantum-cryptography-in-asymmetric-networks
This repository contains example codes used in the paper "Partially device-independent quantum cryptography in asymmetric networks".

The codes are based on the python package [Ncpol2sdpa](https://github.com/peterwittek/ncpol2sdpa), which are no longer maintained. We therefore resort to [PICOS](https://picos-api.gitlab.io/picos/index.html) to generate the SDPs involving block matrices. The resulting SDPs can be efficiently solved using [MOSEK](https://www.mosek.com/), or using built-in solvers at the cost of longer computation time.

## File Descriptions

- **`1sDIQKD_BB84meas.py`**  
  Computes lower bounds on the asymptotic key rates of one-sided device-independent (1SDI) quantum key distribution (QKD) protocol using BB84 measurement settings, where each party performs Pauli-Z and Pauli-X measurements on his/her share of a depolarized Bell state.
  Reproduces Figure 2(a) of the paper.

- **`1sDIQCKA_6statemeas.py`**  
  Computes lower bounds on the asymptotic key rates of 1SDI quantum conference key agreement (QCKA) protocol with six-state measurement settings,  where each party measures Pauli X, Y and Z on his/her share of a depolarized GHZ state. 

- **`1sDIQKD_3DMUB_4bases.py`**  
  Computes lower bounds on the asymptotic key rates of three-dimensional 1SDI-QKD protocol. Each party measures all four mutually unbiased bases (MUBs) on his/her share of a depolarized three-dimensional maximally entangled state.
  Reproduces **Figure 2(b)** of the paper.

- **`1sDI_1outEntr_steeringineq.py`**  and **`2sDI_1outEntr_steeringineq.py`**
  Compute lower bounds on the one-outcome entropy corresponding to the genuine tripartite steering inequalities proposed in [Nat. Commun. 6, 7941 (2015)](https://www.nature.com/articles/ncomms8941).
  Reproduces **Figure 4** of the paper.

- **`1sDI_2outRandomness_steeringineq.py`**
  Computes lower bounds on the two-outcome entropy associated with the genuine tripartite 1SDI steering inequality.
  Reproduces the corresponding curve in **Figure 6(b)** of the paper.

Additional results in the paper can be obtained by modifying these scripts accordingly.
  
