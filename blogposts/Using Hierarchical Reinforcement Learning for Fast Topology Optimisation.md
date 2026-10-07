---
title: "Using Hierarchical Reinforcement Learning for Fast Topology Optimisation"
source: "https://gigatskhondia.medium.com/using-hierarchical-reinforcement-learning-for-fast-topology-optimisation-85aa0c07fb7f"
author:
  - "[[Giorgi Tskhondia]]"
published: 2023-11-25
created: 2026-04-19
description: "Using Hierarchical Reinforcement Learning for Fast Topology Optimisation The below model is a part of Gigala software — open-source software for topology optimisation and engineering design by …"
tags:
  - "clippings"
---
Get unlimited access to the best of Medium for less than $1/week.[Become a member](https://medium.com/plans?source=upgrade_membership---post_top_nav_upsell-----------------------------------------)

[

Become a member

](https://medium.com/plans?source=upgrade_membership---post_top_nav_upsell-----------------------------------------)

The below model is a part of Gigala software — open-source software for topology optimisation and engineering design by reinforcement learning, genetic algorithms and finite element methods.

*‘Topology optimization (TO) is a mathematical method that optimizes material layout within a given design space, for a given set of loads, boundary conditions and constraints with the goal of maximizing the performance of the system. Topology optimization is different from shape optimization and sizing optimization in the sense that the design can attain any shape within the design space, instead of dealing with predefined configurations.’ —* wikipedia

Hierarchical Reinforcement Learning (HRL) is a technique that utilises hierarchies for a better sample efficiency of RL algorithms. Hierarchies can be spatial, temporal or between the abstraction levels. In case of finite element analysis (FEA), hierarchy can be different scales (micro, macro) or it can be different comprehensive levels of mathematical model (e.g. analytical solution — plane stress — 3D model).

In this work, I applied Hierarchical Reinforcement Learning (HRL), \[1\], to topology optimisation of cantilever beam, \[2-3\].

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*Z0yNdTv-YrKz-8_ggKaTdg.png)

Figure 1. Topology obtained by HRL (cherry-picked)

I have modified the codebase from \[4\] to accommodate the topology optimisation task.

One of the main problems when applying reinforcement learning to topology optimisation is scaling up from smaller geometries to bigger design spaces. In this work, I have tried to apply HRL to solve the scale up problem. I have experimented with a 2-level hierarchical agent in 6 by 6, 4 by 4, and 3 by 3 grids. My intention was to speed up the learning.

At first, my reasoning was that an agent at higher level would choose one of the sub sections within the design space grid first, and a low-level agent then would choose one of the cells within the sub-section chosen at higher level. But that would make decisions sequential and interdependent between the hierarchical levels. When I dug into literature, it turned out that most promising HRL algorithms are those who make decision within a hierarchical level relative to other level not sequentially but in parallel, since it greatly speeds up the learning. Making my initial ‘sequential’ views on HRL applications to structural design naïve. Nevertheless, the initial idea of different hierarchical levels doing different things, for example, on a higher level an agent moving to the location on the grid, and on the lower level, the agent changing the topology there, should be investigated further.

When you use a reinforcement learning algorithm which is not a part of robust software like, for example, stable baselines, it is really hard to make it work \[5\]. And apart from solving the main problem of figuring out how to apply RL to the new environment, I was dealing with hyperparameters’ tuning and digging into different code base implementations (mostly ad hoc open source) very deeply. But at the end of the day, this is a kind of work that counts the most.

I found a nice codebase for HRL here \[4\] and built my model on top of it.

There were a few tricks that I used to solve this environment:

- hyperparameter optimisation (HPO)
- tweaking neural network architecture (especially adding dropout layer)
- reward engineering
- making neural network aware of the design space geometry
```c
self.penalty_coeff = 0.10 # 0.25
action=action*(1-self.penalty_coeff*self.x.reshape(len(action),))
```

Penalty coefficient is used to give the agent an ability to do the same actions.

The environment in this RL model is a vector of \[mean density, compliance\], and the action is putting a material into the cell of the grid. The number of actions equals the number of cells in the grid. Time horizon to achieve a subgoal should be less than maximum number of steps in the episode to benefit from abstraction that HRL provides. Currently, I am not sure if this environment vector uniquely maps to a topology.

I am not feeding the topology to the neural network explicitly during the learning, hence I applied the above trick of making NN aware of the design space geometry. It is kind of panelising already taken actions.

This restriction on the action distribution might actually not be the right way to do it, but, even if it is a faulty reasoning that I am making, a neural network can adjust its decisions and make right actions at the end of the day, resulting in an optimal policy that accounts for the bug.Also, it should be noted that relying only on hyperparameters tuning might have overfitted the domain.

Besides, apart from penalising, at first, I wanted to make the neural network to build an understanding of an environment by remembering actions taken (because I did not explicitly input the grid into the neural network, and the neural network did not know any changes happening to the topology). I experimented with adding LSTM in the first layer of actor network as a form of memory, but it really reduced the speed of learning, and potentially required accumulating some number of actions hence mixing up the initial HRL algorithm. So, I abandoned this idea and used the above trick of ‘masking’ (penalising) action distribution instead. But I might return to the LSTM idea later since some papers indicate that entering even one frame of the grid might be sufficient, \[6\].

**Discussion:**

Although the algorithm did not find an optimal topology all the time, I think it is because it is getting into local minima or I did not apply enough learning. I am not sure if it is indigenous to my implementation of the algorithm, or I need to do a better longer HPO.

In terms of performance my implementation of TO with HRL is much slower than my previous work, \[7\], if we count for HPO, but seems to be faster if we do not count for HPO. Another limitation of TO with HRL is that you need to manually set the target state, which is a kind of drawback because you do not know the target state in advance.

The promise that HRL can speed up TO is high. If fully solved, it can allow to go from 2D to 3D topologies with RL.

And although traditional TO will almost always be faster than RL, reinforcement learning can account for a unique constraint where minimising gradients is tough or impossible to do (for nonconvex objectives). Reinforcement learning accounts for these objectives and constraints through the reward function. Besides, generalisability of RL and transfer learning can make RL be faster than classical TO in sum once you train the RL model. Because, one does not need to rerun calculations for each new constraint, but just run RL inference instead.

To sum up, as RL and PINN (that can be used instead of FEA) algorithms become more efficient, we should have a viable competitor to traditional TO algorithms, especially considering AI’s generalisability and transfer of learning. On hardware side analog chips might play a role ([https://www.youtube.com/watch?v=GVsUOuSjvcg](https://www.youtube.com/watch?v=GVsUOuSjvcg) ).

I believe, the idea of hierarchical reinforcement learning can also lead to the systems of the full design cycle, where at the lowest level, the RL agent designs material. One abstraction level above, the RL agent designs a component (assembly of materials), one level above that, the RL agent produces a product (assembly of components). At each level, the RL agent makes higher and higher level design decisions. At the end, an engineer gets her final design that was generated in a hierarchical fashion.

Codebase availability: [https://github.com/gigatskhondia/gigala](https://github.com/gigatskhondia/gigala)

To keep up with the project please visit [https://gigala.io/](https://gigala.io/)

Link to the preprint: [https://www.researchgate.net/publication/376833399\_On\_the\_Quest\_to\_Achieve\_Fast\_Generalizable\_Topology\_Optimization\_with\_Reinforcement\_Learning](https://www.researchgate.net/publication/376833399_On_the_Quest_to_Achieve_Fast_Generalizable_Topology_Optimization_with_Reinforcement_Learning)

**Updated on Apr 2025 with the link to my most recent paper:** [https://www.researchgate.net/publication/389785241\_Reinforcement\_Learning\_Guided\_Engineering\_Design\_from\_Topology\_Optimization\_to\_Advanced\_Modelling](https://www.researchgate.net/publication/389785241_Reinforcement_Learning_Guided_Engineering_Design_from_Topology_Optimization_to_Advanced_Modelling)

**Updated — Sep 26, 2025** **with the link to my latest preprint:** [https://www.researchgate.net/publication/393164291\_Pseudo\_3D\_topology\_optimisation\_with\_reinforcement\_learning](https://www.researchgate.net/publication/393164291_Pseudo_3D_topology_optimisation_with_reinforcement_learning)

\[1\] [https://arxiv.org/abs/1712.00948](https://arxiv.org/abs/1712.00948)

\[2\] [https://www.researchgate.net/publication/360698153\_A\_Tutorial\_on\_Structural\_Optimization](https://www.researchgate.net/publication/360698153_A_Tutorial_on_Structural_Optimization)

\[3\] [https://gigatskhondia.medium.com/topology-optimization-with-reinforcement-learning-d69688ba4fb4](https://gigatskhondia.medium.com/topology-optimization-with-reinforcement-learning-d69688ba4fb4)

\[4\] [https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch](https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch)

\[5\] [https://www.alexirpan.com/2018/02/14/rl-hard.html](https://www.alexirpan.com/2018/02/14/rl-hard.html)

\[6\] [https://arxiv.org/pdf/1507.06527.pdf](https://arxiv.org/pdf/1507.06527.pdf)

\[7\] [https://gigatskhondia.medium.com/topology-optimization-with-reinforcement-learning-d69688ba4fb4](https://gigatskhondia.medium.com/topology-optimization-with-reinforcement-learning-d69688ba4fb4)

[![Giorgi Tskhondia](https://miro.medium.com/v2/resize:fill:96:96/1*7fA2QZN2SZZ-E7R_FvO4Fw.jpeg)](https://gigatskhondia.medium.com/?source=post_page---post_author_info--85aa0c07fb7f---------------------------------------)[15 following](https://gigatskhondia.medium.com/following?source=post_page---post_author_info--85aa0c07fb7f---------------------------------------)

An adept of reinforcement learning and finite element methods. Unlocking the fabric of reality @Gigala