---
title: "Continuous action space reinforcement learning for ‘donut’ topology optimisation"
source: "https://gigatskhondia.medium.com/continuous-action-space-reinforcement-learning-for-donut-topology-optimisation-b109bfa055d1"
author:
  - "[[Giorgi Tskhondia]]"
published: 2025-05-31
created: 2026-04-19
description: "Continuous action space reinforcement learning for ‘donut’ topology optimisation Some portions of text and images were co-created with Gemini On the one hand, the key concept that describes …"
tags:
  - "clippings"
---
Get unlimited access to the best of Medium for less than $1/week.[Become a member](https://medium.com/plans?source=upgrade_membership---post_top_nav_upsell-----------------------------------------)

[

Become a member

](https://medium.com/plans?source=upgrade_membership---post_top_nav_upsell-----------------------------------------)

Some portions of text and images were co-created with Gemini

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*C4faoEj8rVBesHrdOfwJIw.png)

Figure 1. Donut topology to shape to get the final design.

On the one hand, the key concept that describes topological equivalence is homeomorphism. Two shapes are homeomorphic if one can be continuously deformed into the other without tearing, cutting, or gluing. This is why a donut is considered topologically equivalent to a coffee cup (with a handle), as both have exactly one “hole.”

On the other hand, reinforcement learning (RL) is a novel approach to engineering design. It encompasses modern developments in artificial intelligence and safety of classical engineering. For example, in topology optimisation, RL is a learning based, gradient free, generalisable algorithm that can deal with non-convex non-linear design spaces and generally produce more optimal designs than that of by classical optimisation methods (Fig.2).

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*cecg3plzZtnz5LVMpzLayw.png)

Figure 2. Topology obtained by reinforcement learning and finite element methods (5 by 5 grid).

In my work, \[1\], I apply RL to topology optimisation and engineering design with the aim of making these algorithms practical for industrial use. For example, when designing an airplane, the goal is to optimise for both lightness and strength. By creating components with lower mass and higher strength, we directly reduce fuel consumption. This, in turn, leads to lower ticket prices for consumers. My vision outlines how the industrialisation of reinforcement learning based design can achieve these critical outcomes, ultimately benefiting the end-user.

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*7_vkKbkzc9T84lJSu1tVJw.png)

Figure 3. A two dimensional donut with shaping forces applied along its boundaries

If you think of it, topology in Fig.2 and Fig. 3 are homeomorphic, meaning you can get one from the other by shaping (applying loads at different places along torus internal and external boundaries) the topology on Fig.3. We kind of know the area of the final topology in advance (from other experiments), and it is not changing its value during the shaping.

This idea can be realised into continues action space RL for topology optimisation in the following manner. You start from topology on Fig. 3 and by applying loads to it, convert it to topology on Fig. 2. The action space is continues, with the 6 dimensional action vector:

action = \[arc length from the origin out, x-force out, y-force out, arc length from the origin inner, x-force inner, y-force inner\]

where:\-1 ≤ x-force, y-force ≤ 1

0 ≤ arc length ≤ 2\*pi (clockwise )

You can use any specific implementation of RL algorithm for continues action space, for example, as per \[1\] for HRL.

In more complex topologies you could have more than one holes but the over all idea would stay the same. Start with a ‘donut’ and get final topology by shaping it. Potentially it could reduce the compute requirements as you could apply only discrete forces at a few discrete nodes on the circles’ boundaries ( probably, in this case, you would use some kind of splines that dragged the boundaries as well).

Overall, you should consider this article as fun brainstorming of ideas circulating around my work. But, if you want a more scientific treatment of topology optimisation by reinforcement learning please check \[1, 2\].

**Updated — Jun 23, 2025.**

Why do good ideas come to me late at night? Maybe it is because I am constantly thinking on how to speed up topology optimisation by reinforcement learning? Here is the idea, completely unrelated to the above. In FEM analysis you might have different size of the mesh, depending on how nonlinear geometry is locally. Well, I just thought why not to have different size of action elements placing material on the grid. One element is one square, another is three squares, yet another is two squares, etc. Hence you would reduce total number of actions by having ‘various size actions’. And the only thing we would need is to play with ‘action mesh’ to ensure ‘convergence by mesh’. One might use some quick algorithm like e.g. SIMP to roughly locate denser regions for the mesh. Ultimately we would apply ‘bigger actions’ to denser regions and little or no actions to the rest of the grid hence considerably reducing action space.

**Updated — Jun 28, 2025.**

In programs like SolidWorks you can extract 3D surface to a 2D drawing. Hence, you would get three 2D drawings. An interesting idea would be to do topology optimisation by reinforcement learning for these three 2D drawings separately but collectively (each projection would know about the others) in a multi objective optimisation fashion (something like as in *mult\_env\_rl* of \[2\]). This potentially should allow considerably reduce compute requirements because you would substitute 3D optimisation with three 2D optimisations (during the training, you just synthesise online 3D structure from the three 2D drawings and check for optimality).

**Updated — Sep 26, 2025** **with the link to my latest preprint:** [https://www.researchgate.net/publication/393164291\_Pseudo\_3D\_topology\_optimisation\_with\_reinforcement\_learning](https://www.researchgate.net/publication/393164291_Pseudo_3D_topology_optimisation_with_reinforcement_learning)

To keep up with the project please see [https://gigala.io/](https://gigala.io/)

\[1\] [https://www.researchgate.net/publication/389785241\_Reinforcement\_Learning\_Guided\_Engineering\_Design\_from\_Topology\_Optimization\_to\_Advanced\_Modelling](https://www.researchgate.net/publication/389785241_Reinforcement_Learning_Guided_Engineering_Design_from_Topology_Optimization_to_Advanced_Modelling)

\[2\] [https://github.com/gigatskhondia/gigala](https://github.com/gigatskhondia/gigala)

[![Giorgi Tskhondia](https://miro.medium.com/v2/resize:fill:96:96/1*7fA2QZN2SZZ-E7R_FvO4Fw.jpeg)](https://gigatskhondia.medium.com/?source=post_page---post_author_info--b109bfa055d1---------------------------------------)[15 following](https://gigatskhondia.medium.com/following?source=post_page---post_author_info--b109bfa055d1---------------------------------------)

An adept of reinforcement learning and finite element methods. Unlocking the fabric of reality @Gigala