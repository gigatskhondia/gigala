---
title: "Genetic Algorithms pre-training Reinforcement Learning parallel block-refinement strategy for…"
source: "https://gigatskhondia.medium.com/genetic-algorithms-pre-training-reinforcement-learning-parallel-block-refinement-strategy-for-aa5a62c9d6f5"
author:
  - "[[Giorgi Tskhondia]]"
published: 2026-01-09
created: 2026-04-19
description: "Genetic Algorithms pre-training Reinforcement Learning parallel block-refinement strategy for topology optimisation A prerequisite for ‘anchor node’ strategy? I do topology optimisation by …"
tags:
  - "clippings"
---
A prerequisite for ‘anchor node’ strategy?

I do topology optimisation by reinforcement learning research; for fun; in my free time. You can read more about my endeavours in my blog on Medium or in \[1–3\]. The journey is bumpy and with a lot of challenges, but I like it this way.

As I said before, an intriguing speculative idea struck me while watching the NVIDIA Keynote at COMPUTEX 2025, where Jensen Huang mentioned that “we are rendering only one out of ten pixels and guessing the remaining nine” using a technology called DLSS. This concept of selectively computing and inferring the rest inspired me to think on applying a similar principle to topology optimisation. Instead of calculating the entire topology, what if we compute only a subset of key “anchor” nodes and use a neural network to infer the remaining structure?! The combined layout — consisting of anchor and predicted nodes — could then be evaluated using finite element analysis (FEA) to assess structural compliance, providing feedback to guide the optimization process. This approach would have the potential to significantly accelerate learning and improve computational efficiency.

In this article, I wanted to discuss some progress that I am making on the road to my “anchor node” strategy.

I have called my new approach “GA pre-training RL parallel block-refinement” strategy (GA-pt-RL-pbr). Please check my GitHub for more implementation details ([https://github.com/gigatskhondia/gigala/](https://github.com/gigatskhondia/gigala/)). But it goes like this: you divide the grid into blocks. For example, 28 by 28 grid you can divide into 16 equal blocks of 7 by 7 each. Then, first you do GA pre-training on the full 28x28 grid, after that you do RL refinement for each of 7x7 blocks in parallel. You just freeze the rest (taking topology from GA ) and refine one 7x7 block at a time (but in parallel). To do it in parallel you freeze and refine different parts. And the most interesting thing is that if you look at the pictures below it might be just the stuff I was looking for for my so called ‘anchor node’ strategy, \[2\].

Here is a topology after GA pre-training (Fig. 1).

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*2e2VKYqpvJMbZ6loqhj7KQ.png)

Fig.1. Topology after GA pertaining.

And here what I get after the RL refinement stage (Fig. 2).

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*u6tG9MHalMZISCXyjx0AjA.png)

Fig.2. Topology after RL parallel block-refinement.

And finally, if we connect the dots, we get Fig. 3.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*hgS34JlqdB7gu34eg15GlA.png)

Fig. 3. Topology obtained by ‘GA-pt-RL-pbr’ strategy after connecting the dots.

As you can see, topology on Fig.3 resembles the topology on Fig. 1.More broadly, the ‘GA pre-training RL parallel block-refinement’ strategy can potentially be used to find topologies for bigger design spaces or to train generative models where, in terms of machine learning, you have data (Fig.2) and target (Fig.1 or Fig.3) for different boundary conditions. In fact, the nodes on (Fig. 3) are positioned at places of minimum *compliance,* hence the structure might be seen as a light weight product of generative design.

To sum up, I just wanted to say that my research on topology optimisation by reinforcement learning is not just optimal shapes. It is forming brains for agent-builders and agent-technologists. After an agent learns an optimal design, it has the brain (neural network) to implement that design in physical world i.e. to build a house, a bridge, or a road in a sequential manner. Applying my methods to robotics, where my algorithms provide the brain of the agent-builder, opens whole new possibilities for future constructions.

Codebase availability: [https://github.com/gigatskhondia/gigala](https://github.com/gigatskhondia/gigala)

To keep up with the project please visit [https://gigala.io/](https://gigala.io/)

\[1\] [https://www.researchgate.net/publication/389785241\_Reinforcement\_Learning\_Guided\_Engineering\_Design\_from\_Topology\_Optimization\_to\_Advanced\_Modelling](https://www.researchgate.net/publication/389785241_Reinforcement_Learning_Guided_Engineering_Design_from_Topology_Optimization_to_Advanced_Modelling)

\[2\] [https://www.researchgate.net/publication/393164291\_Pseudo\_3D\_topology\_optimisation\_with\_reinforcement\_learning](https://www.researchgate.net/publication/393164291_Pseudo_3D_topology_optimisation_with_reinforcement_learning)

\[3\] [https://www.researchgate.net/publication/398406554\_Practical\_topology\_optimization\_with\_deep\_reinforcement\_learning](https://www.researchgate.net/publication/398406554_Practical_topology_optimization_with_deep_reinforcement_learning)