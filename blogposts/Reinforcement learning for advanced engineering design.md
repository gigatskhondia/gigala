---
title: "Reinforcement learning for advanced engineering design"
source: "https://gigatskhondia.medium.com/reinforcement-learning-for-advanced-engineering-design-e386be4461fa"
author:
  - "[[Giorgi Tskhondia]]"
published: 2025-05-28
created: 2026-04-19
description: "Reinforcement learning for advanced engineering design Reinforcement learning (RL) is a novel approach to engineering design. It encompasses modern developments in artificial intelligence and safety …"
tags:
  - "clippings"
---
Get unlimited access to the best of Medium for less than $1/week.[Become a member](https://medium.com/plans?source=upgrade_membership---post_top_nav_upsell-----------------------------------------)

[

Become a member

](https://medium.com/plans?source=upgrade_membership---post_top_nav_upsell-----------------------------------------)

Reinforcement learning (RL) is a novel approach to engineering design. It encompasses modern developments in artificial intelligence and safety of classical engineering. For example, in topology optimisation, RL is a learning based, gradient free, generalizable algorithm that can deal with non-convex non-linear design spaces and generally produce more optimal designs than that of by classical optimisation methods (Fig.1).

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*cecg3plzZtnz5LVMpzLayw.png)

Figure 1. Topology obtained by reinforcement learning and finite element methods (5 by 5 grid).

In my work, \[1\], I apply RL to topology optimisation and engineering design with the aim of making these algorithms practical for industrial use. Let’s say you are designing an airplane. You need to make it light and strong. For this you design components that have lower mass and higher strength. The lighter the weight, the less the fuel consumption. The less the fuel consumption the less the cost for the airplane ticket. Hence, there is a direct impact of these algorithms to the consumer. From this perspective, I wanted to share my vision on how this industrialisation of RL based design could be accomplished.

To do something you need an end goal. To do something great, you need an unrealistic end goal. For me this unrealistic end goal is to make ‘an RL and finite element based system’ that would design machinery for us to travel to distant stars. Now, to make it more manageable, I have to set a more realistic goal: I want to create a system (a number of algorithms) that could be able to design e.g. a rocket engine end to end, and here is how I intend to do this.

First of all, we would need some kind of hierarchical system to design an engine in the bulk. This hierarchical system would have multiple abstraction levels. On the lowest level, it could design a material, one abstraction level above that, it could design a component, one level above that, it could design an assemble of components, and, finally, at the highest level, it would design an entire engine. All these levels should be able to communicate with each other during the training process. For example, they could combine reward from different abstraction levels in some form.

In addition, when you have different optimisation objectives, you might experiment with e.g. *Vectorized Environments* from *Stable-Baselines3* or try building your custom hierarchical algorithm*.* Just imagine, one objective is mechanics, another one is electrodynamics, yet another one is magnetodynamics. You might try to use Vectorized Environments to optimise for all of them at the same time, and maybe find some Nash equilibrium of objectives for the final design. Some experiments with *mult env* can be found in \[2\].

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*x2nJNxP0B1vh1VDY7Ff9Bw.png)

Figure 2. Two environments optimised simultaneously with slightly different reward functions by utilising Vectorized Environments from Stable-Baselines3.

Also, to make the approach more practical you would need RL algorithms to become more sample efficient. For this, I intend to: utilise GPUs ( more specifically, I intend to rewrite my code by using Mac Metal); try model-based RL ( not only MCTS but also I experiment with genetic algorithms to be the model of the world that allows you do planning); apply asynchronous algorithms that could reduce run time of the algorithm as e.g. in \[3\]; employ hybrid algorithms ( for example, I have experimented with combining genetic algorithms and RL).Another interesting idea came to me while watching Nvidia Keynote at COMPUTEX 2025, where Jensen Huang said that ‘we are rendering only one out of ten pixels, and guessing the rest of the nine pixels’. They are using something called DLSS for that ‘guessing’. So, I thought why not to try guessing topology optimisation ‘pixels’?! We can calculate only ‘anchor’ nodes in the topology and make neural network draw the rest of the topology. Then we would check ‘anchor nodes + guessed nodes’ with FEA and understand the compliance of the structure, thus guiding the topology optimisation process. Apparently, it could speed up the learning.

We could also need a breakthrough in RL to make my approach truly practical, but I think there is plenty of ideas to try first even without the need for the breakthrough.

As a side note, I would like to add that RL is very tricky. It is prone to instability by a lot, and if you do not use mainstream algorithms like PPO from SB3, the conclusion of RL’s validity is not that obvious and straightforward. Nevertheless the promise of this approach is very high ( considering its generalisability, transfer learning, and learning based nature).

To imbibe my vision you can start from my blog on Medium and to understand RL paradigm in depth you can address \[4\]. If you are a graduate student or a PhD researcher you can even try choosing this approach as your thesis, and, who knows, maybe it is going to be you who makes a significant contribution — in fact, I am sure, it is going to be you:)

**Updated — Sep 26, 2025** **with the link to my latest preprint:** [https://www.researchgate.net/publication/393164291\_Pseudo\_3D\_topology\_optimisation\_with\_reinforcement\_learning](https://www.researchgate.net/publication/393164291_Pseudo_3D_topology_optimisation_with_reinforcement_learning)

To keep up with the project please see [https://gigala.io/](https://gigala.io/)

\[1\] [https://www.researchgate.net/publication/389785241\_Reinforcement\_Learning\_Guided\_Engineering\_Design\_from\_Topology\_Optimization\_to\_Advanced\_Modelling](https://www.researchgate.net/publication/389785241_Reinforcement_Learning_Guided_Engineering_Design_from_Topology_Optimization_to_Advanced_Modelling)

\[2\] [https://github.com/gigatskhondia/gigala](https://github.com/gigatskhondia/gigala)

\[3\] [https://arxiv.org/abs/1602.01783](https://arxiv.org/abs/1602.01783)

\[4\] [https://spinningup.openai.com/en/latest/](https://spinningup.openai.com/en/latest/)

[![Giorgi Tskhondia](https://miro.medium.com/v2/resize:fill:96:96/1*7fA2QZN2SZZ-E7R_FvO4Fw.jpeg)](https://gigatskhondia.medium.com/?source=post_page---post_author_info--e386be4461fa---------------------------------------)[15 following](https://gigatskhondia.medium.com/following?source=post_page---post_author_info--e386be4461fa---------------------------------------)

An adept of reinforcement learning and finite element methods. Unlocking the fabric of reality @Gigala