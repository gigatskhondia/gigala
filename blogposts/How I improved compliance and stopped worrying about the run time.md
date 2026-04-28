---
title: "How I improved compliance and stopped worrying about the run time"
source: "https://gigatskhondia.medium.com/how-i-improved-compliance-and-stoped-worrying-about-the-run-time-b986617de75b"
author:
  - "[[Giorgi Tskhondia]]"
published: 2025-11-23
created: 2026-04-19
description: "How I improved compliance and stopped worrying about the run time when doing topology optimisation by reinforcement learning Topology optimisation is a computational design method that finds the most …"
tags:
  - "clippings"
---
Get unlimited access to the best of Medium for less than $1/week.[Become a member](https://medium.com/plans?source=upgrade_membership---post_top_nav_upsell-----------------------------------------)

[

Become a member

](https://medium.com/plans?source=upgrade_membership---post_top_nav_upsell-----------------------------------------)

when doing topology optimisation by reinforcement learning

Topology optimisation is a computational design method that finds the most efficient distribution of material within a defined design space, subject to specific loads and boundary conditions. It starts with a large, solid block of material (or void ) and an algorithm then iteratively removes (or adds) all the material that (does not) significantly contribute to the part’s stiffness and strength, forcing the structure to carry the load along the most direct paths. The result is an organic, lightweight, and high-performance geometry (often resembling bone or natural structures) that maximises the part’s structural efficiency, leading to significant weight and material savings, especially when combined with Additive Manufacturing (3D printing).

I have been doing research in applying reinforcement learning to topology optimisation since 2018. During this time, I have investigated Proximal Policy Optimisation; Hierarchical Reinforcement Learning; combination of genetic algorithms and RL; AlphaZero algorithm; and Pseudo 3D topology optimisation in order to make this approach practical for industry needs, \[1–2\].

In this article, I am going to explain one neat trick that I just came up with to improve compliance metrics and reduce run time of the algorithm. Namely, for 6 by 6 grid, I was able to reduce compliance from 19.92 to 18.16 (the lower the better), and reduce algorithm’s execution time from 19.337 min to 12.954 ( and even to 10.796) min (Fig.1 — Fig.2).

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*4NxOQOD4c8L7pzqf5DeO3Q.png)

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*9fgH5WRzF_kfUXyXA_D55w.png)

Figure 1. Topologies obtained by reinforcement learning: the usual way (on the left); by using the smart skipping trick (on the right).

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*raCexlsfLDWeEmIx9IG_Yw.png)

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*V5IMgn0_FirzxUg_MifFkg.png)

Figure 2. Topologies obtained by reinforcement learning: via using the smart skipping trick.

The idea of an method is as follows: for the first 300k (or 100k) iterations I do FEA to calculate compliance, but then I start comparing the current state (topology) with the best state (of best compliance) by cosine similarity and if it is less than threshold I do not do FEA and assign reward of zero, but if cosine similarity greater than threshold I do FEA and calculate the compliance in a usual way. The assumption is that better states are close to each other after some number of iterations.Snapshot of the smart skipping trick can be seen below (for full implementation details please check my codebase).

```c
# version 1
if  Tr.global_count <= 300_001:
    tmp, const = fast_stopt(self.args, self.x)
    self.reward = (1/tmp)**2
    if self.reward > Tr.max_reward:
        Tr.max_reward = self.reward
        Tr.best_state = self.x.reshape(6*6)
else:
    if cosine_similarity(self.x.reshape(6*6), Tr.best_state) < Tr.threshold:
        self.reward = 0
        const = 0
    else:
        tmp, const = fast_stopt(self.args, self.x)
        self.reward = 10*(1/tmp)**2
```
```c
# version 2
if Tr.global_count <= 100_001:
    tmp, const = fast_stopt(self.args, self.x)
    self.reward = (1/tmp)**2
else:
    if cosine_similarity(self.x.reshape(6*6), Tr.best_state) < (0.5+Tr.global_count/(2*ts))*Tr.threshold:
        self.reward = 0                                                            
        const = 0
    else:
        tmp, const = fast_stopt(self.args, self.x)
        self.reward = (1/tmp)**2

if self.reward > Tr.max_reward:
    Tr.max_reward = self.reward
    Tr.best_state = self.x.reshape(6*6)
```

Also I experimented with sparse rewards which reduced run time down to ~7min for 6 by 6 grid.

```c
# version 3 - sparse rewards
if done:
    tmp, const = fast_stopt(self.args, self.x)
    
    a1=1/tmp
    a2=1/const
    
    self.reward = 2*a1*a2/(a1+a2)
```

It also allowed to get the following topologies for 8 by 8, 9 by 9, and 10 by 10 grids with pure DRL (Fig.3 — Fig.5):

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*uiL_uaBVgywAN5OndtrA7g.png)

Figure 3. Topology obtained by pure deep reinforcement learning with sparse rewards (8 by 8 grid; wall-clock time is 9.06 min; 1M iterations).

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*m3LMl23UMv31U6Lefv9wdg.png)

Figure 4. Topology obtained by pure deep reinforcement learning with sparse rewards (9 by 9 grid; wall-clock time is 17.56 min; 2M iterations).

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*iZGj8PR36vc2nnsGm-XFpA.png)

Figure 5. Topology obtained by pure deep reinforcement learning with sparse rewards (10 by 10 grid; wall-clock time is 26.33 min; 3M iterations).

That is more states than there are stars in the Universe! And all of this was done under 30 min wall-clock run time on my Mac M1 CPU.

Finally, I optimised 12x12 grid with sparse rewards, requesting ‘one island’,requesting ‘smooth’ geometry, and smart post-processing (removing elements with less than 2 neighbours), Fig. 6.

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*ZzBtN6eTBS5Lmy-twrqaqA.png)

Figure 6. Topology obtained by pure deep reinforcement learning with sparse rewards, requesting ‘one island’, requesting ‘smooth’ geometry, and smart post-processing for 12 by 12 grid (wall-clock time is ~35 min; 4M iterations)

Preprint: [https://www.researchgate.net/publication/398406554\_Practical\_topology\_optimization\_with\_deep\_reinforcement\_learning](https://www.researchgate.net/publication/398406554_Practical_topology_optimization_with_deep_reinforcement_learning)

Codebase availability: [https://github.com/gigatskhondia/gigala](https://github.com/gigatskhondia/gigala)

To keep up with the project please visit [https://gigala.io/](https://gigala.io/)

\[1\] [https://www.researchgate.net/publication/389785241\_Reinforcement\_Learning\_Guided\_Engineering\_Design\_from\_Topology\_Optimization\_to\_Advanced\_Modelling](https://www.researchgate.net/publication/389785241_Reinforcement_Learning_Guided_Engineering_Design_from_Topology_Optimization_to_Advanced_Modelling)

\[2\] [https://www.researchgate.net/publication/393164291\_Pseudo\_3D\_topology\_optimisation\_with\_reinforcement\_learning](https://www.researchgate.net/publication/393164291_Pseudo_3D_topology_optimisation_with_reinforcement_learning)

[![Giorgi Tskhondia](https://miro.medium.com/v2/resize:fill:96:96/1*7fA2QZN2SZZ-E7R_FvO4Fw.jpeg)](https://gigatskhondia.medium.com/?source=post_page---post_author_info--b986617de75b---------------------------------------)[15 following](https://gigatskhondia.medium.com/following?source=post_page---post_author_info--b986617de75b---------------------------------------)

An adept of reinforcement learning and finite element methods. Unlocking the fabric of reality @Gigala