### Leaky Integrated-and-Fire (LIF) Neuron
SNNs communicate using discrete electrical spikes 

## biological neuron 
dendrites receive electrical signals through synapses -> 
membrane potential voltage changes -> 
neuron's membrane is leaky -> 
gradual decays if no signals arrive -> 
neuron generates enough potential -> 
generates action potential (spike) at axon hilllock -> 
spike travesl down axon to communicate with other neuron -> 
refractory period after membrane potential 


## LIF model 
- membrane potential V increases with input current 
- potential "leaks" decays over time
- neuron fires a spike when V reaches threshold 


## Parameters / Initializations 
V[t]: membrane potential at time t 
dt: time step 
R: resistance to flow of electrical signasl (set to 1)
I[t]: input current at time t


## LIF Neuron Equation

The Leaky Integrate-and-Fire (LIF) neuron is described by:

$$
\tau_m \frac{dV(t)}{dt} = -[V(t) - V_{rest}] + R_m I(t)
$$

- $V(t)$: Membrane potential at time $t$
- $V_{rest}$: Resting membrane potential
- $\tau_m$: Membrane time constant ($\tau_m = R_m C_m$)
- $R_m$: Membrane resistance
- $I(t)$: Input current at time $t$

**Discrete (simulation) version:**

$$
V[t+1] = V[t] + \frac{dt}{\tau_m}(-[V[t] - V_{rest}] + R_m I[t])
$$

When $V$ reaches a threshold, the neuron fires a spike and $V$ is reset.

