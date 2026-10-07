---
title: "Topology optimisation by genetic algorithms pre-training and reinforcement learning refinement."
source: "https://gigatskhondia.medium.com/topology-optimisation-by-genetic-algorithms-pre-training-and-reinforcement-learning-refinement-b31854aa9625"
author:
  - "[[Giorgi Tskhondia]]"
published: 2024-12-07
created: 2026-04-19
description: "Topology optimisation by genetic algorithms pre-training and reinforcement learning refinement. The below model is a part of Gigala software — open-source software for topology optimisation and …"
tags:
  - "clippings"
---
Get unlimited access to the best of Medium for less than $1/week.[Become a member](https://medium.com/plans?source=upgrade_membership---post_top_nav_upsell-----------------------------------------)

[

Become a member

](https://medium.com/plans?source=upgrade_membership---post_top_nav_upsell-----------------------------------------)

The below model is a part of Gigala software — open-source software for topology optimisation and engineering design by reinforcement learning, genetic algorithms and finite element methods.

Topology optimisation is a mathematical method which spatially optimises the distribution of material within a defined domain, by fulfilling given constraints previously established and minimising a predefined cost function.

I have been trying to approach the problem of topology optimisation separately by reinforcement learning and genetic algorithms, \[1–5\].

With the reinforcement learning I was successfully able to optimise a 6 by 6 grid. The reinforcement learning (RL) produced better results (in terms of a compliance metric) for 6 by 6 grid than genetic algorithms for the comparable wall clock run times. However, when I ran RL for larger grids it was incapable of producing optimal design within adequate times. So I needed a way to speed up and improve the performance of my algorithms.

In this work, I combined genetic algorithm (GA) and reinforcement learning to optimise a 10 by 10 grid in the following manner:

1. I have run fast GA (for about 1 min wall clock time), where actions were placing an element in the void on the grid (void is 0 and an element is 1).
2. I have set an action space for the RL problem as removal of boundary elements obtained in (1) only (Figure 1). All other actions ( grid elements) were not considered in RL problem. Hence I considerably reduced combinatorial space for the RL task.
3. I ran RL PPO algorithm for about 32 min wall clock time.
4. Experiments showed that GA pre-training + RL refinement was better than GA alone (in terms of a compliance metric) for the same wall clock run times. And, obviously, GA+RL was better than RL alone.
![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*Y8hdlopK2ufsIddYYGRNQw.png)

Figure 1. Topology with boundary layer (action layer) in black, internal material in yellow and empty space in red

I trained my model on Apple M1 Pro. I have used DEAP framework for the GA part. And, I have used *stable\_baselines3* and *gym* for PPO algorithm’s implementation for the RL part.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*ltKLmbsv572vsCtSF3jdcA.png)

Figure 2. Action space model

I applied downward load at the bottom right corner and fixed an element on the left side alike cantilever beam. Objective function was a compliance in topology optimisation sense i.e. an inverse strength of a material. RL improved compliance metric of GA from 19.3654 to 18.2786 (the lower the better).

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*Z97jFdeaup40Z8TOBqR2Cw.png)

Figure 3. Topology obtained by GA pre-training

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*jWNviPy_8nDdpw2kKlKADQ.png)

Figure 4. Topology obtained by RL refinement

**Results:** Experiments show that GA pre-training + RL refinement is better than GA alone (in terms of a compliance metric) for the same wall clock run times. And, GA+RL is better than RL alone.Modelling of this kind gives hope to apply my algorithms for a larger design spaces producing optimal topologies within an adequate wall clock time.

Codebase for the model can be find in \[6\].

To keep up with the project please visit [https://gigala.io/](https://gigala.io/)

**Update on Apr 2025 with the link to my most recent paper:** [https://www.researchgate.net/publication/389785241\_Reinforcement\_Learning\_Guided\_Engineering\_Design\_from\_Topology\_Optimization\_to\_Advanced\_Modelling](https://www.researchgate.net/publication/389785241_Reinforcement_Learning_Guided_Engineering_Design_from_Topology_Optimization_to_Advanced_Modelling)

**Updated — Sep 26, 2025** **with the link to my latest preprint:** [https://www.researchgate.net/publication/393164291\_Pseudo\_3D\_topology\_optimisation\_with\_reinforcement\_learning](https://www.researchgate.net/publication/393164291_Pseudo_3D_topology_optimisation_with_reinforcement_learning)

\[1\] [https://gigatskhondia.medium.com/engineering-design-by-reinforcement-learning-and-finite-element-methods-82eb57796424](https://gigatskhondia.medium.com/engineering-design-by-reinforcement-learning-and-finite-element-methods-82eb57796424)

\[2\] [https://gigatskhondia.medium.com/engineering-design-by-genetic-algorithms-and-finite-element-methods-5077ebadd16e](https://gigatskhondia.medium.com/engineering-design-by-genetic-algorithms-and-finite-element-methods-5077ebadd16e)

\[3\] [https://gigatskhondia.medium.com/topology-optimization-with-reinforcement-learning-d69688ba4fb4](https://gigatskhondia.medium.com/topology-optimization-with-reinforcement-learning-d69688ba4fb4)

\[4\] [https://gigatskhondia.medium.com/using-hierarchical-reinforcement-learning-for-fast-topology-optimisation-85aa0c07fb7f](https://gigatskhondia.medium.com/using-hierarchical-reinforcement-learning-for-fast-topology-optimisation-85aa0c07fb7f)

\[5\] [https://gigatskhondia.medium.com/formulating-engineering-design-as-full-reinforcement-learning-problem-871b6d594239](https://gigatskhondia.medium.com/formulating-engineering-design-as-full-reinforcement-learning-problem-871b6d594239)

\[6\] [https://github.com/gigatskhondia/gigala](https://github.com/gigatskhondia/gigala)

\[7\] Preprint [https://www.researchgate.net/publication/387602607\_Topology\_optimization\_by\_genetic\_algorithms\_pre-training\_and\_reinforcement\_learning\_refinement](https://www.researchgate.net/publication/387602607_Topology_optimization_by_genetic_algorithms_pre-training_and_reinforcement_learning_refinement)

[![Giorgi Tskhondia](https://miro.medium.com/v2/resize:fill:96:96/1*7fA2QZN2SZZ-E7R_FvO4Fw.jpeg)](https://gigatskhondia.medium.com/?source=post_page---post_author_info--b31854aa9625---------------------------------------)[15 following](https://gigatskhondia.medium.com/following?source=post_page---post_author_info--b31854aa9625---------------------------------------)

An adept of reinforcement learning and finite element methods. Unlocking the fabric of reality @Gigala