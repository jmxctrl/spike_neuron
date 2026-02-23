this readme is intended for personal stream of thoughts https://hackmd.io/@S6bykdDqSNCB7m5UZ34uTw/SkW6dFLdZl

## basic neuroscience 
main components: 
- dendrites 
- soma (cell body)
- axon 
- axon terminal 
- synaptic cleft 
- receptor (excitatory or inhibitory)
- recycling 

1. dendrites receive chemical signal (signal causes small voltage change) -> `I_synaptic`
2. soma sums up all dendrite inputs at time t (v >= 55mV, fires) -> `dV/dt = -(V - V_rest) * R * (I_external + I_synaptic)`
3. axon generates electrical spike (all or nothing) -> `if >= threshold: spikes[t, i] = 1`
4. spike reaches synapses causing neurotransmitter release -> `weight multiplication`
5. neurotransmitter cross the gap -> 
6. bind to receptors on the next neuron's dendrite -> `sum(W[i, j])`
    - excitatory synapse: +0.1 to +1 mV per spike 
    - inhibitory synapse: -0.5 to -2 mV per spike 

## The general SNN engine: 
* encoder / decoder 
* weight matrix 
* synaptic transmission 
* spiking 
* STDP (incoming)
   1. calculate time difference Δt > 0 = strengthen Δt < 0 = weaken 
   2. calcualte magnitude 
   3. aply magnitude to learning rate 
   4. apply sign (direction)
   5. update weight 
* temporal control (incoming) 


## STDP Mathematical Breakdown

### 1. Time Difference ($\Delta t$)
The time difference determines if a connection should be strengthened or weakened:

$$\Delta t = t_{\text{post}} - t_{\text{pre}}$$

- **$\Delta t > 0$**: Post fires AFTER pre → Causal → Strengthen connection (LTP)
- **$\Delta t < 0$**: Post fires BEFORE pre → Not causal → Weaken connection (LTD)

**Example:**
- Pre-synaptic neuron spikes at $t = 10\text{ms}$
- Post-synaptic neuron spikes at $t = 15\text{ms}$
- $\Delta t = 15 - 10 = +5\text{ms}$ → Strengthen

### 2. Exponential Decay Function

The magnitude of weight change decays exponentially with time:

$$e^{-|\Delta t|/\tau}$$

**Breaking it down:**
- $e$ = Euler's number $\approx 2.718$ (natural base for continuous growth/decay)
- $|\Delta t|$ = absolute value of time difference
- $\tau$ (tau) = time constant (controls decay speed)
- Negative exponent = decay (larger $|\Delta t|$ → smaller magnitude)

**Properties:**
- When $\Delta t = 0$: $e^0 = 1.0$ (maximum effect)
- When $|\Delta t| = \tau$: $e^{-1} \approx 0.37$ (drops to 37%)
- When $|\Delta t| = 2\tau$: $e^{-2} \approx 0.14$ (drops to 14%)
- When $|\Delta t| = 5\tau$: $e^{-5} \approx 0.007$ (nearly zero)

### 3. Complete STDP Formula

**For $\Delta t > 0$ (LTP - Long-Term Potentiation):**

$$\Delta W = A_+ \times e^{-\Delta t/\tau_+}$$

(more negative exponent -> smaller value -> less strengthening)
- requires pre-synaptic neuron to fire first (aka pre helped cause post to fire -> stronger connection)

**For $\Delta t < 0$ (LTD - Long-Term Depression):**

$$\Delta W = -A_- \times e^{\Delta t/\tau_-}$$

(Δt negative -> smaller value -> less weakening)
- requires post synaptic neuron to fire first, pre didn't help as much --> weaker connection 


**Parameters:**
- $A_+$ = maximum weight increase (e.g., 0.01)
- $A_-$ = maximum weight decrease (e.g., 0.012)
- $\tau_+$ = LTP time constant (e.g., 20ms)
- $\tau_-$ = LTD time constant (e.g., 20ms)

### 4. Numerical Example

**Given:**
- $A_+ = 0.01$, $A_- = 0.012$
- $\tau_+ = \tau_- = 20\text{ms}$
- Pre-synaptic spike at $t = 10\text{ms}$
- Post-synaptic spike at $t = 15\text{ms}$

**Calculation:**
1. $\Delta t = 15 - 10 = +5\text{ms}$ (LTP case)
2. Magnitude = $e^{-5/20} = e^{-0.25} \approx 0.7788$
3. $\Delta W = 0.01 \times 0.7788 \approx 0.0078$
4. New weight: $W_{\text{new}} = W_{\text{old}} + 0.0078$ 


## Definitions  
Weight Matrix `W[i,j]` 
* i = post-synaptic neuron 
* j = pre-synaptic neuron 
* W[3, 5] = connection from neuron 5 to neuron 3 

Spike Array spikes `spikes[t, i]`
* t = timestep in ms 
* i = neuron index 
* Value = 1 True 0 False 
* spikes[10, 5] = 1 means neuron 5 fired at t = 10 ms 


## Neuromodulation 
Classical neurotransmitters like Glutamate and GABA create the response itself. While neuromodulators modify those responses by analyzing 
1. speed 
2. intensity 
3. duration 
4. learning 
