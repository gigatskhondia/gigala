---
title: "Using Hierarchical Reinforcement Learning for End to End Rocket Engine Design"
source: "https://gigatskhondia.medium.com/using-hierarchical-reinforcement-learning-for-end-to-end-rocket-engine-design-5efbb3282d9d"
author:
  - "[[Giorgi Tskhondia]]"
published: 2025-11-08
created: 2026-04-19
description: "Using Hierarchical Reinforcement Learning for End to End Rocket Engine Design All in one paradigm - press the button and get the result? My recent years passion is applying reinforcement learning to …"
tags:
  - "clippings"
---
Get unlimited access to the best of Medium for less than $1/week.[Become a member](https://medium.com/plans?source=upgrade_membership---post_top_nav_upsell-----------------------------------------)

[

Become a member

](https://medium.com/plans?source=upgrade_membership---post_top_nav_upsell-----------------------------------------)

All in one paradigm - press the button and get the result?

My recent years passion is applying reinforcement learning to engineering design of mechanical elements and components \[1–2\]. Starting from 2018, I am gradually but stubbornly researching this topic in the hope of unlocking the illimitable potential of artificial intelligence for the real world engineering applications.

On my bumpy road, I have investigated Proximal Policy Optimisation; Hierarchical Reinforcement Learning; combination of genetic algorithms and RL; AlphaZero algorithm; and even Pseudo 3D topology optimisation in order to make this approach practical for industry needs (Fig. 1 and Fig. 2).

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*YYHRlmtj0vxI7BGMNcCeFg.png)

Fig.1. Cantilever topology optimisation by reinforcement learning, \[1\]

But now I’m aiming for something bigger.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*qerRB1orQuxxmp8augyLUw.png)

Fig.2. Pseudo 3D topology optimisation with reinforcement learning, \[2\]

Believe it or not, I genuinely think that it is totally feasible to design a rocket engine end to end with my approach. What we need is physics simulator ( in the form of finite element models and rocket trajectory simulator, e.g. \[3\]), hierarchical reinforcement learning (potentially combined with genetic algorithms), and a personal computer (not sure that PC’s power is enough for this task, but at least it is how I am going to do this).

Using hierarchical reinforcement learning, where you have different abstraction levels or execution hierarchies of a design, with ‘managers’ (who do planning) and ‘workers’ who do the ‘actual work’, can lead to the systems of the full design cycle, where at the lowest level, the RL agent e.g. designs material. One abstraction level above, the RL agent designs a component (assembly of materials), one level above that, the RL agent produces a product (assembly of components).

At each level, the RL agent makes design decisions within its abstraction level, but all the levels communicate with each other during the training. At the end, the final design is produced in an end to end hierarchical fashion.

For the sake of example, let’s imagine we apply three abstraction levels to design our engine.

On level # 0, we would design individual components of the engine, like nozzle, for instance (Fig.3), by optimising their shape with reinforcement learning.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*vRZHM5tgJyGyw6BsM5G39Q.png)

Fig. 3. Nozzle of the rocket (picture taken from the Internet)

On level # 1, we would combine individual components into propulsion system, and optimise for the better placement of constituent elements, (Fig.4).

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*7Ywkr0UK1uSI4s0r-MVjzw.png)

Fig. 4. Assembling the rocket engine, \[4\]

Finally, on level # 2, we would test our design within the framework of entire rocket, for vibration, for example ( Fig.5).

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*R8IVt5eek_pXdOMnitD7ng.png)

Fig.5. Integration testing of our engine within the framework of entire rocket, \[4\]

Importantly, all of these three abstraction levels would be designed simultaneously in parallel by a single RL agent in hierarchical fashion. The three levels would communicate with each other and produce the design end to end (e.g. as in \[5\]).Early works of applying HRL to topology optimisation can be found in \[1–2\].

Some initial experiments on nozzle design can be seen on Fig. 6 and in my codebase (see below).

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*-unkJJj0UX71etfgCJfmiQ.png)

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*3xTHKYH_ut3IocpB1M7pHw.png)

Fig.6. Experiments on nozzle design: Bell Nozzle (on the left) and Spike Nozzle (on the right).

I do not know how to you, but to me this approach seems to be within reach and opens the door to space exploration. Now, would you press the button and get the result?

**Updated Jan 29, 2026 with the link to most recent article:** [https://www.sciencepublishinggroup.com/article/10.11648/j.ajasr.20261201.11](https://www.sciencepublishinggroup.com/article/10.11648/j.ajasr.20261201.11)

Codebase availability: [https://github.com/gigatskhondia/gigala](https://github.com/gigatskhondia/gigala)

To keep up with the project please visit [https://gigala.io/](https://gigala.io/)

\[1\] [https://www.researchgate.net/publication/389785241\_Reinforcement\_Learning\_Guided\_Engineering\_Design\_from\_Topology\_Optimization\_to\_Advanced\_Modelling](https://www.researchgate.net/publication/389785241_Reinforcement_Learning_Guided_Engineering_Design_from_Topology_Optimization_to_Advanced_Modelling)

\[2\] [https://www.researchgate.net/publication/393164291\_Pseudo\_3D\_topology\_optimisation\_with\_reinforcement\_learning](https://www.researchgate.net/publication/393164291_Pseudo_3D_topology_optimisation_with_reinforcement_learning)

\[3\] [https://github.com/RocketPy-Team/RocketPy](https://github.com/RocketPy-Team/RocketPy)

\[4\] Rocket Propulsion Elements, George P. Sutton and Oscar Biblarz

\[5\] [https://arxiv.org/pdf/1712.00948](https://arxiv.org/pdf/1712.00948)

[![Giorgi Tskhondia](https://miro.medium.com/v2/resize:fill:96:96/1*7fA2QZN2SZZ-E7R_FvO4Fw.jpeg)](https://gigatskhondia.medium.com/?source=post_page---post_author_info--5efbb3282d9d---------------------------------------)[15 following](https://gigatskhondia.medium.com/following?source=post_page---post_author_info--5efbb3282d9d---------------------------------------)

An adept of reinforcement learning and finite element methods. Unlocking the fabric of reality @Gigala