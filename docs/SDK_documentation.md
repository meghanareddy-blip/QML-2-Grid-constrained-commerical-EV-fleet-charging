# SDK Integration Documentation — QML-02 Grid-Constrained EV Fleet Charging

**Track I: Quantum Machine Learning (QML-02)**  
**Frameworks Used:** QpiAI QCloud (`qpiai-quantum`) & QpiAI Opt (`qpiai-opt`)

---

## 1. System Architecture & Dual-Layer Design

Our solution implements a dual-layer hybrid quantum-classical pipeline addressing transformer feeder capacity limits ($150\text{ kW}$) and dynamic Time-of-Use tariffs:

* **Layer 1: Quantum Risk Parameterization (`qpiai-quantum`)**
  * Constructs a 1-qubit Parameterized Quantum Circuit (PQC).
  * Encodes the vehicle's arrival State-of-Charge ($\text{SoC}$) onto the Bloch sphere using an $R_y(\theta)$ rotation gate ($\theta = \pi \times \frac{\text{SoC}}{100}$).
  * Computes the Pauli-$Z$ expectation value $\langle Z \rangle = P(|0\rangle) - P(|1\rangle)$ to derive the Quantum Departure Risk Factor ($\alpha_i$).

* **Layer 2: QUBO Feeder Optimization (`qpiai-opt`)**
  * Formulates a 24-slot fleet charging schedule into a QUBO / Ising Hamiltonian energy matrix.
  * Inputs quantum departure risk factors ($\alpha_i$), Time-of-Use electricity tariffs, and departure deadlines.
  * Solves slot allocation while strictly enforcing the $150\text{ kW}$ transformer load ceiling.

---

## 2. API Setup & Authentication Configuration

* **QCloud Console:** `https://qcloud.qpiai.tech`
* **Opt Console:** `https://opt.qpiai.tech`
* **Environment Configuration (`backend/.env`):**
  ```env
  QPIAI_QUANTUM_API_KEY="your_qcloud_api_token"
  QPIAI_OPT_API_KEY="your_opt_api_token"
  HOST="127.0.0.1"
  PORT=8000