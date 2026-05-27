---
title: "Reinforcement learning guided engineering design"
source: "https://gigatskhondia.medium.com/reinforcement-learning-guided-engineering-design-dc83a3abb7f7"
author:
  - "[[Giorgi Tskhondia]]"
published: 2025-02-18
created: 2026-04-19
description: "Reinforcement learning guided engineering design The below model is a part of Gigala software — open-source software for topology optimisation and engineering design by reinforcement learning …"
tags:
  - "clippings"
---
Get unlimited access to the best of Medium for less than $1/week.[Become a member](https://medium.com/plans?source=upgrade_membership---post_top_nav_upsell-----------------------------------------)

[

Become a member

](https://medium.com/plans?source=upgrade_membership---post_top_nav_upsell-----------------------------------------)

The below model is a part of Gigala software — open-source software for topology optimisation and engineering design by reinforcement learning, genetic algorithms and finite element methods.

Rapid development of artificial intelligence facilitates engineering design by creating advanced tooling to assist an engineer. The pace of computer technological development dictates a modern designer to equip herself with at least basic understanding of current artificial intelligence algorithms and methods.From this perspective, this article is an attempt to provide some guidelines to artificial intelligence aided engineering design with the focus on topology optimisation by reinforcement learning.

Topology optimisation (TO)got traction in recent years and is incorporated in almost all main commercial CAD systems nowadays. Optimisation approaches used by these systems are mostly based on genetic or gradient optimisation combined with finite element methods. But little attention has been paid to using reinforcement learning (RL) for topology optimisation. Only few scientific groups have attempted to apply RL to topology design so far. Main advantages of RL include being gradient-free approach and its generalisability. Whereas the disadvantages are sample inefficiency of RL and scalability problem for huge finite element meshes.

Likewise*,* domain conceptual design has a lot of value for creating unique machinery and cherishing the future design methods. Some hope that domain design can be solved with large language models (LLM) like GPT, which relies on reinforcement learning from human feedback (RLHF), or Deepseek which relies on chain of thought (CoT) and mixture of experts (MoE). These LLMs show ability to generate programming code that compiles and is logically correct. The same LLMs can be used to generate a ‘language’ or ‘code’ that describes the geometry of mechanical structures in parametric models, component diagrams, structural diagrams, etc. This ‘code’ would represent unique structures and designs. One example of this could be LLM producing the input file for finite element model, then one would run FE analysis with this input file via utilising a FE engine (not LLM), and then the produced output were again fed to LLM for post-processing.

Additionally,some diffusion probabilistic models that create photorealistic images can be used for an engineer’s inspiration. Here, for generating sketches of a design one needs to provide a prompt using natural language only. These sketches can later be refined with detailed drawings or fed to a multimodal LLMs for further design tailoring.

**Focus of this article**

Learning how to optimise topology has major importance in structural design for aerospace, automotive, offshore, and other industries. Despite outstanding successes in topology optimisation methods, \[1\], complexity of design space continues to increase as the industry faces new challenges. A viable alternative to conventional topology optimisation methods might be deep reinforcement learning approach, \[2\]. Reinforcement learning offers gradient free, learning based, generalisable topology optimisation paradigm suitable for non-convex design spaces. It can be seen as a smart search in solution space.

Deep reinforcement learning has had great success in artificial intelligence applications. Among them, beating the champion of the game of Go in 2016, mastering many Atari games, \[3\], and optimising the work of data centres. In my work, I combine deep reinforcement learning, genetic algorithms (GA), and finite element analysis (FEA) for the purpose of topology optimisation. I have experimented with 10x10, 6x6, 5x5 and 4x4 FE grids, tested generalizability, applied simple reward function for the RL agent, applied density field-based topology optimization, and used simpler input features’ vector compared to other works. I have experimented with PPO, \[4\], HRL \[5\], and GA algorithms. Also, my code implementation is scalable and robust.

The RL agents were able to find optimal topologies (more optimal than by gradient methods) for a wide range of boundary conditions and applied loads.

In topology optimisation settings, the finite element model represents an environment to which an agent applies actions and from which it gets observations and rewards. An agent uses neural network to decide on its actions. Actions change topology and the new topology is then subjected to finite element analysis (FEA). Finite element analysis produces the state, which then is fed to neural network. And the process repeats itself. The agent gets rewards if it meets an optimisation objective of minimising compliance. The end result of the modelling (after inference stage) is an optimised topology. The inference stage is a usual greedy inference where an agent makes actions of altering the topology based on observations only.

I argue that engineering design in general and topology optimisation in particular should be formulated as full reinforcement learning but not as just a deterministic problem where the next state depends on previous state and action alone but not also on random noise.

For the purpose of my argument, I will use terms dynamic programming and reinforcement learning interchangeably as per \[6\]. In its simple form, dynamic programming can be described as solution to the equation of full reinforcement learning (Figure 1):

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*2Y_pyBHnZjdQvPCcV9pb2Q.png)

Figure 1. Equation of full reinforcement learning (1)

I will also use words engineering design and topology optimisation interchangeably to some extent. However, my argument can be understood in a broader sense when applied to all forms of engineering design but not only to topology optimisation. For instance, I will touch such processes as metal stamping and designing computer chips layouts.

Let’s start from mechanical engineering and metal pressure treatment. In topology optimisation, especially 3D printed structures, you can have a large range of fatigue allowable stresses that depend on the thermal history experienced by the material in each point. Basically it means that your component will experience fatigue based on history and ordering of the structure’s 3D printing. To some degree, fatigue initiates at random after certain amount of cycles of exploited component. Here we can say that actions of producing the structure in a sequence, whether during design or actual manufacturing, affect the outset of fatigue of the component in the future. In terms of equation (1), if *wk* is random outset of fatigue, it should depend on N-2(nd) state and N-2 (nd) action*.* Or more broadly on all previous states and actions in a sequence in a sense that N-2(nd) depend on N-3(d) and so on down to state number zero. Hence, equation of full reinforcement learning (1) can be applied to both manufacturing and design (as a way for future manufacturing steps) of the component.

Another example is metal stamping where metal stamping mold’s shape would depend on the topology of the component being produced. The mold would install different micro defects into the component after metal pressure treatment (stamping). These defects will be installed at random. Think *wk* is micro defects. Again, defects would depend on the topology of the component, on the shape of the mold. These defects affect metal fracture in the future component exploitation, and might even have non-stationary evolution that leads to fatigue. Hence, equation (1) can be used and the problem should be formulated as full reinforcement learning.

Now, let’s say a few words about electrical engineering and chip design. If you design chips you need to deal with something called parasitic capacitance (an unavoidable and usually unwanted capacitance that exists between the parts of an electronic component or circuit simply because of their proximity to each other). This makes transition dynamics ( how the voltage and current in different parts of the circuit change over time in response to different inputs) stochastic and to some degree random. Once more, we have here *wk* asparasitic capacitance. The layout of the chip you design would affect implicitly the parasitic capacitance you get in the future. Hence, again, the chip design can be seen as a full reinforcement learning problem here.

**Experiments**

Using codebase from \[7\], I have tried to rework its topology optimisation approach by replacing its conventional gradient based optimisation method with reinforcement learning and genetic algorithms.

Topology optimisationaims to distribute material in a design space such that it supports some fixed points or “normals” and withstands a set of applied forces or loads with the best efficiency. To illustrate how we can formulate this, I concentrate on one of design problems from \[8\], describing elementally discretised design domain as per (Figure 2).

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*xXWcAp5mooRuXRdVdTGgyQ.png)

Figure 2. 2D representation of an elementally discretized starting topology of a 8 × 8 cantilever beam

The large grey rectangle here represents the design space. The load force, denoted by the downwards-pointing arrow, is being applied at the bottom right corner. There are fixed points here as well, which are at the top and bottom left corners of the design space corresponding to a normal force from some external rigid joint (Figure 2).

There are many ways to choose the arrangement of these finite elements. The simplest one is to make them square and organise them on a rectangular grid.

I have applied PPO implementation from *stable baselines 3* library and reworked HRL implementation from \[9\]. For both PPO and HRL, I have created custom *OpenAI Gym* environments.

The action space consisted of N² actions of ‘filling void space (actually with the density of 1e-4) with an element (density of 1)’, where N by N is dimension of the grid.

I have experimented with 6 by 6 grid for PPO, and 4 by 4 grid for HRL.

For PPO setting, I have tested generalisability of the model by randomising the places of force application across different episodes and being different in training and inference. The environment for this setting was the topology on the grid.

I experimented with squared rooted and squared reciprocals of compliance as reward functions. I trained my model for 1M steps (for about 80min of wall-clock time) on 2,9 GHz Dual-Core Intel Core i5 computer, for PPO. And for 5000 steps on Apple M1 Pro, for HRL.

For HRL setting, hyperparameters (including network architecture) were obtained by *optuna.* For PPO settings, the exact architecture is as per *stable baselines 3* implementation.

The results of applying PPO and HRL to topology optimisation of Cantilever beam is presented in Figure 3 below.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*cuLHkCNb80D9nmfXRx617A.png)

I have experimented with a 2-level hierarchical agent in 4 by 4 grid and did not test generalizability for HRL setting. My intention with HRL was to speed up the learning.

RL was able to reliably find optimal topologies for a range of different boundary conditions and applied forces. In terms of performance my implementation of HRL is much slower than PPO if we count for HPO (hyper parameter optimization), but seems to be faster if we do not count for HPO. Another limitation of HRL is that you need to manually set the target state, which is a kind of drawback because you do not know the target state in advance and only can set it by experimentation.

Number of calculation steps is much less than the number of possible combinations on the grid, hence the neural network is not merely learning by heart the most optimal solution (Figure 4).

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*QxHIJFpjgx5ujo9GlTMyKg.png)

Figure 4: Episode reward for PPO setting

In other set of experiments, I have combined genetic algorithms and reinforcement learning in a straightforward sequential (one method after another) way. First, I apply genetic algorithms to get an outline of the topology and then I ‘fine-tune’ or refine obtained topology by reinforcement learning approach. In this way, I can obtain both more optimal topologies and reduce wall-clock time. I was able to optimise topology for 10 by 10 grid, which can be seen as an improvement over 6 by 6 topology obtained by reinforcement learning alone. It should be mentioned that genetic algorithm alone was not able to produce such an optimal topology as by combination of reinforcement learning and genetic algorithms in adequate wall clock times.

I combined genetic algorithm (GA) and reinforcement learning to optimise a 10 by 10 grid in the following manner:

1\. I have run fast GA (for about 1 min wall-clock time), where actions were placing an element in the void on the grid (void is 0 and an element is 1).

2\. I have set an action space for the RL problem as removal of boundary elements obtained in (1) only (Figure 5). All other actions (grid elements) were not considered in RL problem. Hence, I considerably reduced combinatorial space for the RL task.

3\. I ran RL PPO algorithm for about 32 min wall clock time.

4\. Experiments showed that GA pre-training + RL refinement was better than GA alone (in terms of a compliance metric) for the same wall clock run times. And, obviously, GA+RL was better than RL alone.

I trained my model on Apple M1 Pro. I have used DEAP framework for the GA part. And, I have used *stable\_baselines3* and *gym* for PPO algorithm’s implementation for the RL part.

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*TZs5iq4tLwE8WV3Qq-cN2w.png)

Figure 5. Topology with boundary layer (action layer) in black, internal material in yellow and empty space in red

I applied downward load at the bottom right corner and fixed an element on the left side alike cantilever beam (Figure 2). Objective function was a compliance in topology optimisation sense i.e. an inverse strength of a material. RL improved compliance metric of GA from 19.3654 to 18.2786 (the lower the better). Optimal topologies are depicted in Figure 6 below.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*wSbwJ9K_b-1KQDZ-tAqa1g.png)

Learning progress for RL refinement can be seen in Figure 7.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*moiKdQXhpyA_LJn3N1Dipw.png)

Figure 7: Episode reward for PPO refinement

**Discussion**One of the main problems when applying reinforcement learning to topology optimisation is scaling up from smaller geometries to larger design spaces. In this work, I have tried to apply HRL, PPO and GA to solve the scale up problem. My intention was to speed up the learning hence potentially to be able to solve larger topologies, especially with non-convex objectives.

As stated earlier, HRL is a technique that utilises hierarchies for a better sample efficiency of RL algorithms. Hierarchies can be spatial, temporal or between the abstraction levels. In the case of finite element analysis, hierarchy can be different scales (micro, macro) or it can be different comprehensive levels of mathematical model (e.g. analytical solution — plane stress — 3D model).

There were a few tricks that I used to solve the HRL environment:

· hyperparameter optimisation (HPO; done with *optuna*)

· tweaking neural network architecture (especially adding dropout layer)

· reward engineering

· making neural network aware of the design space geometry (Figure 8)

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*x5OhpWI27UWdOsyDMD_2iQ.png)

Figure 8. Code snippet for HRL setting

Penalty coefficient is used to give the agent an ability to do the same actions (Figure 8).

The continuous environment in this HRL model is a vector of \[mean density, compliance\], and the action is putting a material into the cell of the grid. The number of actions equals the number of cells in the grid. Time horizon to achieve a subgoal should be less than maximum number of steps in the episode to benefit from abstraction that HRL provides. It should be noted that the environment vector is not uniquely maps to a topology.

I am not feeding the topology to the neural network explicitly during the learning, hence I applied the above trick of making neural network ‘aware’ of the design space geometry (Figure 8). It is kind of applying penalty to already taken actions. This restriction on the action distribution might actually not be the optimal way to do it, but, even if it is a faulty reasoning, a neural network can adjust its decisions and make right actions, resulting in an optimal policy that accounts for the ‘bug’.

Also, it should be noted that relying only on hyperparameters tuning might have overfitted the domain.

Apart from applying the penalty, at first, I wanted to make the neural network to build an understanding of an environment by remembering actions taken (because I did not explicitly input the grid into the neural network, and the neural network did not know any changes happening to the topology). I experimented with adding LSTM in the first layer of actor network as a form of memory, but it reduced the speed of learning by a large margin, and potentially required accumulating some number of actions hence mixing up the initial HRL algorithm. So, I abandoned this idea and used the above trick of ‘masking’ (applying penalty) action distribution instead. But I might return to the LSTM idea later since some papers indicate that entering even one frame of the grid might be sufficient for ‘remembering’ \[10\].

Both PPO and HRL produced optimal topologies.

By number of states, I mean number of combinations of cells in a grid.

The promise that RL can speed up topology optimisation is high. If fully solved, it can allow to go from 2D to 3D topologies with RL especially for larger design spaces and non-convex objectives. Other ways to scale/speed up to larger topologies might be progressive refinement method \[8\], where one gradually increases the complexity of design space without requiring additional computation, and asynchronous deep reinforcement learning \[11\] that uses asynchronous gradient descent for optimization of deep neural network controllers and enables half the training time on a single multi-core CPU instead of a GPU. In the future, speeding up the calculation might be accomplished: by making neural networks ‘smaller’ via using sparsity, quantization, or distillation techniques; by applying physics informed neural networks ([https://en.wikipedia.org/wiki/Physics-informed\_neural\_networks](https://en.wikipedia.org/wiki/Physics-informed_neural_networks)) in place of finite element methods; by using evolutionary strategies \[12\] that can run in parallel; or, on the hardware side, by using analog chips ([https://www.youtube.com/watch?v=GVsUOuSjvcg](https://www.youtube.com/watch?v=GVsUOuSjvcg)).

I believe, the idea of hierarchical reinforcement learning can also lead to the systems of the full design cycle, where at the lowest level, the RL agent designs material. One abstraction level above, the RL agent designs a component (assembly of materials), one level above that, the RL agent produces a product (assembly of components). All it is done concurrently. At each level, the RL agent makes higher and higher level design decisions. At the end, an engineer gets her final design that was generated in a hierarchical fashion. As for the effectiveness of current implementation of hierarchical RL, it was estimated that for each iteration there were ~11 actions that the agent was making. For 4500 iterations (min number of iteration at which we can get optimal solution), it is 11\*4500 = 49500 actions. On the other hand, for 4 by 4 grid we have 2¹⁶=65536 possible states (combination of cells). Hence, accounting for the fact that from that 11 actions that the agent was making some were repetitive, HRL is much more effective than a brute force approach ( not in terms of wall clock time for 4 by 4 grid, but in terms of number of operations).

Although traditional TO will almost always be faster than RL in wall clock time meaning, reinforcement learning can account for a unique constraint where minimising gradients is tough or impossible to do (for nonconvex objectives). Reinforcement learning accounts for these objectives and constraints through the reward function. Besides, generalisability of RL and transfer learning can make RL be more compute efficient than classical TO in total once you train the RL model. Because, one does not need to rerun calculations for each new constraint, but just run RL inference instead. Additionally, every time you run inference after RL training, it gives you a slightly different topology. Hence, RL can be leveraged at inference time as it produces not a single design but several different designs from which you can choose the best. It happens since at each inference time a neural network fires slightly different weights thus producing a slightly different design.

Other experiments show that GA pre-training + RL refinement is better than GA alone (in terms of a compliance metric) for the same wall clock run times. And, GA+RL is better than RL alone.

Modelling of this kind gives hope to apply my algorithms for a larger design spaces producing optimal topologies within an adequate wall-clock time.

Additionally, let’s try to estimate practicality of the proposed approach if a supercomputer is available. First, let’s try to estimate a supercomputer’s compute capacity. In the article ([https://research.google/blog/chip-design-with-deep-reinforcement-learning/](https://research.google/blog/chip-design-with-deep-reinforcement-learning/)), on the macro placements of Ariane picture, there are ~ (32 by 34) grid with ~6 different colouring. That means that there are 6¹⁰⁸⁸ states that can be handled by Google supercomputer in an adequate wall clock time. On the other hand, we have a binary grid in topology optimization. Hence to understand how many nodes we can process, we need to solve the following equation: 6¹⁰⁸⁸=2^x, from which it follows that x~2812 elements. However, it should be noted that the google paper \[13\] used only reinforcement learning and did not apply finite element methods in their algorithm. I instead used combination of RL and genetic algorithm (that can handle more elements in less time) and applied finite element methods. Let’s assume that these two factors compensate each other (in terms of calculation time) and we are still at 2812 elements that can be handled by a supercomputer. From the literature it was reported that gradient topology optimisation can handle up to 30000 elements max, but nevertheless gradient topology optimisation would not provide such optimal results as a combination of RL and GA presented in this paper.

Potential practical application of topology optimisation with RL for small grids can be microscopic topology optimisation, \[14\], and metamaterials design, \[15\].

Finally, an open research question in topology optimisation is how to make the produced designs suitable for a wide range of manufacturing methods but not only for additive manufacturing and 3D printing. I propose to approach this issue by guiding topology optimisation with reinforcement learning by techniques from AI safety research. For example, by learning a reward function from human feedback (favouring the designs that are manufacturable) and then to optimise that reward function, \[16\].

Some benchmarking can be found in the paper: [https://jngr5.com/index.php/journal-of-next-generation-resea/article/view/95](https://jngr5.com/index.php/journal-of-next-generation-resea/article/view/95)

Replication of Results: Code from this paper is available at [https://github.com/gigatskhondia/gigala](https://github.com/gigatskhondia/gigala)

**Updated Jan 29, 2026 with the link to most recent article:** [https://www.sciencepublishinggroup.com/article/10.11648/j.ajasr.20261201.11](https://www.sciencepublishinggroup.com/article/10.11648/j.ajasr.20261201.11)

To keep up with the project please visit [https://gigala.io/](https://gigala.io/)

**Literature**

\[1\] Andreassen, E., Clausen, A., Schevenels, M., Lazarov, B. S., and Sigmund, O. Efficient topology optimization in matlab using 88 lines of code. Structural and Multidisciplinary Optimization,43(1):1–16, 2011.

\[2\] Richard Sutton and Andrew Barto. Reinforcement Learning: An Introduction. Second Edition, MIT Press, Cambridge, MA, 2018

\[3\] Playing Atari with Deep Reinforcement Learning, Mnih et al, 2013

\[4\] Proximal Policy Optimization Algorithms, Schulman et al, 2017.

\[5\] Learning Multi-Level Hierarchies with Hindsight, Andrew Levy et al, 2019

\[6\] Dynamic programming and optimal control by D.P. Bertsekas

\[7\] A Tutorial on Structural Optimization, Sam Greydanus, May 2022, DOI: 10.48550/arXiv.2205.08966

\[8\] Deep reinforcement learning for engineering design through topology optimization of elementally discretized design domains, Nathan K. Brown et al, [https://doi.org/10.1016/j.matdes.2022.110672](https://doi.org/10.1016/j.matdes.2022.110672)

\[9\] @misc{pytorch\_hac, author = {Barhate, Nikhil}, title = {PyTorch Implementation of Hierarchical Actor-Critic}, year = {2021}, publisher = {GitHub}, journal = {GitHub repository}, howpublished ={\\url{https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch}},}

\[10\] Deep Recurrent Q-Learning for Partially Observable MDPs, Matthew Hausknecht and Peter Stone, 2017

\[11\] Asynchronous Methods for Deep Reinforcement Learning, Volodymyr Mnih et al, 2016

\[12\] Evolution Strategies as a Scalable Alternative to Reinforcement Learning, Tim Salimans et al., 2017

\[13\] A graph placement methodology for fast chip design, Azalia Mirhoseini et al, Nature volume 594, pages207–212 (2021)

\[14\] Microscopic stress-constrained two-scale topology optimisation for additive manufacturing, Xiaopeng Zhang et al, January 2025Virtual and Physical Prototyping 20(1)

\[15\] Deep reinforcement learning for the design of mechanical metamaterials with tunable deformation and hysteretic characteristics, Nathan K. Brown et al, Materials & Design Volume 235, November 2023, 112428

\[16\] Deep reinforcement learning from human preferences, Paul Christiano et al, arXiv:1706.03741

[![Giorgi Tskhondia](https://miro.medium.com/v2/resize:fill:96:96/1*7fA2QZN2SZZ-E7R_FvO4Fw.jpeg)](https://gigatskhondia.medium.com/?source=post_page---post_author_info--dc83a3abb7f7---------------------------------------)[15 following](https://gigatskhondia.medium.com/following?source=post_page---post_author_info--dc83a3abb7f7---------------------------------------)

An adept of reinforcement learning and finite element methods. Unlocking the fabric of reality @Gigala