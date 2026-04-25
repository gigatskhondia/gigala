## Gigala (Engineering design by reinforcement learning, genetic algorithms and finite element methods)

Gigala is an open-source AI-native framework for intelligent engineering design of physical structures — combining finite element analysis with reinforcement learning and genetic algorithms to produce optimal designs.

Started in 2018, Gigala is one of the earliest implementations of RL-based topology optimization — predating most academic work in this space. The framework has since grown into a research platform with clear applications in structural mechanics, offshore engineering, and space construction.

Reinforcement learning brings a fundamentally different approach to topology optimization: global, gradient-free, non-convex, and generalizable — capable of handling stochastic loading and practical engineering constraints. Its sequential nature makes it naturally extensible to manufacturing process planning (agent-technologist) and autonomous assembly of complex structures (agent-builder), including lunar base construction under dynamic loading.

**Current milestone:**  hybrid GA-RL outperforming pure DRL at 32×32 resolution (effective 64×64 via symmetry exploitation).
Target application: autonomous structural assembly for lunar base programs.

Modules:
* Topology optimization (structural mechanics)
* Offshore pipelay dynamics → now in [Ocean Intella](https://github.com/gigatskhondia/ocean_intella) 
 
Design philosophy:
* Open source — fully customizable, no black boxes
* Open access — no license fees, no institutional barriers
* Runs on your laptop — practical performance, low carbon footprint
* Python-native — built for the research community
* AI-first — not bolted on, built in

Cite this work:
* [Reinforcement Learning Guided Engineering Design: from Topology Optimization to Advanced Modelling](https://jngr5.com/index.php/journal-of-next-generation-resea/article/view/95)
* [Practical Topology Optimization with Deep Reinforcement Learning and Genetic Algorithms](https://www.sciencepg.com/article/10.11648/j.ajasr.20261201.11)

→ [Blog](https://gigatskhondia.medium.com/) · [ResearchGate](https://www.researchgate.net/profile/Giorgi-Tskhondia) 

RL agent designing a cantilever:
![Cantilever design by RL](https://github.com/user-attachments/assets/ae471032-56eb-4907-9f0b-e7a7d30038b9)

→ Follow the project: [Gigala](https://gigala.io/)

If Gigala is useful to you:

- ⭐ Star this repo
- [Sponsor](https://www.paypal.me/gigatskhondia) the project
- [Get in touch](https://gigala.io/) — feedback and collaborations welcome
