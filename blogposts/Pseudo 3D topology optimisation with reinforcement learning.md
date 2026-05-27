---
title: "Pseudo 3D topology optimisation with reinforcement learning"
source: "https://gigatskhondia.medium.com/pseudo-3d-topology-optimisation-with-reinforcement-learning-2bed7b1118e8"
author:
  - "[[Giorgi Tskhondia]]"
published: 2025-06-29
created: 2026-04-19
description: "Pseudo 3D topology optimisation with reinforcement learning Paving the way to practical RL guided topology optimisation The below model is a part of Gigala software — open-source software for …"
tags:
  - "clippings"
---
Get unlimited access to the best of Medium for less than $1/week.[Become a member](https://medium.com/plans?source=upgrade_membership---post_top_nav_upsell-----------------------------------------)

[

Become a member

](https://medium.com/plans?source=upgrade_membership---post_top_nav_upsell-----------------------------------------)

Paving the way to practical RL guided topology optimisation

The below model is a part of Gigala software — open-source software for topology optimisation and engineering design by reinforcement learning, genetic algorithms and finite element methods.

**Topology optimization (TO)** is a computational approach aimed at determining the optimal material layout within a given design space to maximize performance under specified loads, boundary conditions, and constraints. It is widely used in various engineering domains, including structural, fluid, acoustic, and thermal design. Several methodologies exist for implementing TO, such as homogenization-based methods, density-based methods, evolutionary algorithms, and boundary variation techniques. These approaches typically rely on the **finite element (FE) method**, which is well-suited for modelling complex geometries and nonlinear behaviours.

In my work, I introduced a topology optimization approach that integrates the FE method with **deep reinforcement learning**, producing novel and efficient topologies, \[1\].

Long standing goal, however, was to make my algorithms practical for industrial use and now I believe I might have a glimpse on how to do just that.

The first trick that can be applied is that of breaking the topology optimisation task to ‘scale hierarchies’. For example, in hierarchical reinforcement learning (HRL) terms, you can have ‘workers’ designing sub-topologies and a ‘manager’ assembling these sub-topologies together in a bigger topology. In this way, at each abstraction level, your RL agent would take only ‘lower dimension‘ actions compared to the case of designing the entire topology in one go.

Another intriguing idea struck me while watching the NVIDIA Keynote at COMPUTEX 2025, where Jensen Huang mentioned that “we are rendering only one out of ten pixels and guessing the remaining nine” using a technology called DLSS. This concept of selectively computing and inferring the rest inspired me to try to apply a similar principle to topology optimization. Instead of calculating the entire topology, what if we compute only a subset of key “anchor” nodes and use a neural network to infer the remaining structure? The combined layout — consisting of anchor and predicted nodes — could then be evaluated using finite element analysis (FEA) to assess structural compliance, providing feedback to guide the optimization process. This approach would have the potential to significantly accelerate learning and improve computational efficiency.

Third, in finite element analysis (FEM), mesh size often varies depending on the degree of geometric nonlinearity in different regions. This gave me the idea: why not apply the same principle to the action space in topology optimization? Instead of using a uniform grid where each action places material in a single square, we could allow actions of varying sizes — some affecting one square, others spanning two, three, or more. This would reduce the total number of actions needed by enabling **multi-scale actions**. To maintain accuracy, we would adjust the “action mesh” to ensure convergence with respect to mesh resolution (the actual FE mesh could, however, stay uniform). A fast algorithm like SIMP could be used initially to identify high-density regions, which would then receive larger actions. Meanwhile, sparse or low-density areas would require smaller or no actions. This strategy could significantly reduce the size of the action space while preserving optimization quality.

Fourthly, in other set of experiments, I have combined genetic algorithms and reinforcement learning in a straightforward sequential (one method after another) way. First, I apply genetic algorithms to get an outline of the topology and then I ‘fine-tune’ or refine obtained topology by reinforcement learning approach. In this way, I obtained both more optimal topologies and reduced wall-clock time. I was able to optimize topology for 10 by 10 grid, which can be seen as an improvement over 6 by 6 topology obtained by reinforcement learning alone \[1, 2\].

Finally, in CAD programs like SolidWorks, it’s possible to extract 3D surfaces into 2D projections, typically resulting in three orthogonal views. This inspired an idea: perform topology optimization using reinforcement learning on these three 2D projections separately, but in a coordinated, multi-objective manner — where each 2D agent is aware of the others. This concept is prototyped in *multi\_env\_rl* of \[1\]. The key advantage is that, instead of directly optimizing in 3D (which is computationally intensive), we can perform three parallel 2D optimizations. During training, a 3D structure could be reconstructed on-the-fly from the three 2D outputs and evaluated for compliance or other objectives (To reconstruct the 3D structure, you simply take the intersection of the extrusions from the three 2D projections). This strategy could significantly reduce computational requirements while still achieving effective 3D topology optimization( e.g. full 3D optimization contains 2¹⁰⁰⁰ states on a 10 by 10 by 10 grid, my method would contain only 2³⁰⁰ states. And for full 3D optimization you would need 1000 actions, and for my method the three independent environments would only need 100 actions each). Some experiments on multi objective optimization as well as effective prototype on pseudo 3D topology optimisation with RL can be found in \[1\] and on Fig.1 and Fig.2 below.

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*3bFQGlC9UUZY04yXcSdNbQ.png)

Figure 1. Experiments on multi env topology optimisation with reinforcement learning, \[1\]

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*LfOOJD7FJ9p6sbs7mspPXw.png)

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*t5cblqiTDlWDnUuZuX52UA.png)

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*yfQoF7iIK93YQeKbIJNVaw.png)

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*HbTT_JwH4EyES9IGEf0rlg.png)

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*cLTrawQMmbO4Z7E14Jpqmg.png)

Figure 2. Experiments on pseudo 3D topology optimisation with reinforcement learning for 4x4x4 grid, \[1\]

Snippet forcustom-made *stable\_baselines3* environment used during the modelling can be seen below:

```c
class CrossRewardEnv(gymnasium.Env):
    def __init__(self):
        super().__init__()
        self.env1 = CantileverEnv()
        self.env2 = CantileverEnv()
        self.env3 = CantileverEnv()

        self.observation_space = spaces.Dict({
            'X_projection': self.env1.observation_space,
            'Y_projection': self.env2.observation_space,
            'Z_projection': self.env2.observation_space,
        })

        self.action_space = spaces.MultiDiscrete([x0*y0, x0*y0, x0*y0])
        self.step1_=0
        
    def reset(self,seed=0):
        obs1, info1 = self.env1.reset()
        obs2, info2 = self.env2.reset()
        obs3, info3 = self.env3.reset()
        self.step1_=0
        return {
            'X_projection': obs1,
            'Y_projection': obs2,
            'Z_projection': obs3
        }, {
            'X_projection': info1,
            'Y_projection': info2,
            'Z_projection': info3
        }

    def step(self, action):
        a1 = action[0]
        a2 = action[1]
        a3 = action[2]

        obs1_, r1, done1,_, info1 = self.env1.step(a1)
        obs2_, r2, done2,_, info2 = self.env2.step(a2)
        obs3_, r3, done3,_, info3 = self.env3.step(a3)
        obs1=obs1_.reshape(y0,x0).astype(np.uint8)
        obs2=obs2_.reshape(y0,x0).astype(np.uint8)
        obs3=obs3_.reshape(y0,x0).astype(np.uint8)
        topology = reconstruct_3d_structure(obs1, obs2, obs3)
        obs1=obs1_
        obs2=obs2_
        obs3=obs3_

        self.coord, self.elcon, bc_node  = get_voxel_edge_pairs(topology)
        
        arr = np.array(self.elcon)
        
                        
        bc_val=[0]*len(bc_node)
        global_force=[0, 0, -10, 0, 0, 0] + list([0 for i in range(6*(np.max(arr)+1)-6)])
                
        try:
            d, compliance, t_length  = FEA_u(self.coord, self.elcon, bc_node, bc_val, global_force)
        except:
            compliance, t_length = 1e9, x0**4
        
        self.step1_+=1
        
        a1 = 1/compliance
        s_u = np.sum(topology)
        a2 = s_u/x0**3
        
        reward=2*a1*a2/(a1+a2)
        
        self.env1.ext_reward=reward
        self.env2.ext_reward=reward
        self.env3.ext_reward=reward
        done = done1 or done2 or done3 or bool(s_u>0.5*x0**3)
        return {
            'X_projection': obs1,
            'Y_projection': obs2,
            'Z_projection': obs3
        }, reward, done, False, {
            'X_projection': info1,
            'Y_projection': info2,
            'Z_projection': info3,
        }

    def render(self, mode='human'):    
        draw(self.coord, self.elcon,'red')   

    def close(self):
        self.env1.close()
        self.env2.close()
        self.env3.close()
```

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*bcTs64a4AfZt0KvKmuUagg.png)

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*1Os0MxTiH3-6u1QZMCJX5A.png)

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*1TdMC1w5qsDOB5V_imAPVw.png)

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*YUt59Yzs8-OAvs4F_UURtg.png)

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*H5r9F8QXJHMh-YWNAfeifg.png)

Figure 3. Experiments on pseudo 3D topology optimisation with reinforcement learning for 5x5x5 grid, \[1\]

As you can see, my work explores several novel ideas for accelerating and enhancing topology optimization using deep reinforcement learning and finite element analysis. Inspired by techniques from graphics rendering, adaptive meshing in FEM, and projection-based design in CAD software, the proposed framework introduces:

- A method for inferring full topologies by calculating only a sparse set of “anchor” nodes and predicting the rest with a neural network.
- A variable-size action approach that reduces the action space by adapting action granularity to local structural density.
- A multi-objective 2D projection-based strategy that replaces full 3D optimization with coordinated optimization of three 2D views, reconstructing the 3D topology from their intersecting extrusions.

Together, these approaches have the potential to significantly reduce computational cost while maintaining, or even improving, design quality.Some wording for this article was enhanced by chatGPT.

If you are interested in my research please check my latest papers, \[2–3\]

**Updated on Aug 18, 2025:**

Fixed some boundary conditions in my work. Now nice and symmetrical topology is obtained by pseudo 3D topology optimisation with reinforcement learning.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*4dfoHoVjWKtVscAuNvg4Ww.png)

Figure 4. Topology obtained by pseudo 3D topology optimisation with reinforcement learning for 4x4x4 grid.

**Updated on Oct 2, 2025:**

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*gWgeLaioVU7bSmWDXLZ81w.png)

Figure 5. Topology obtained by pseudo 3D topology optimisation with genetic algorithms for 3D solid element of 4x4x8 size.

**Updated on Oct 5, 2025:**

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*LLZ47osCaiiEYptjpL2ADQ.png)

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*3bTN3DaWcKj97wXUuD3fkQ.png)

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*Ft-guzcS6iv7S30TRHaBPw.png)

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*dysPlkr6TBh95z6BdOVTUA.png)

Figure 6. 3D solid element cantilever obtained by pseudo-3D topology optimisation with genetic algorithms of 4x4x8 mesh

**Updated on Oct 21, 2025:**

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*Dq_T8AOMkdsyms57B5BGFA.png)

Figure 7. The 3x3x5 cantilever obtained by pseudo-3D topology optimisation with reinforcement learning

**Updated Jan 29, 2026 with the link to most recent article:** [https://www.sciencepublishinggroup.com/article/10.11648/j.ajasr.20261201.11](https://www.sciencepublishinggroup.com/article/10.11648/j.ajasr.20261201.11)

To keep up with the project please see [https://gigala.io/](https://gigala.io/)

\[1\] [https://github.com/gigatskhondia/gigala](https://github.com/gigatskhondia/gigala)

\[2\] [https://www.researchgate.net/publication/389785241\_Reinforcement\_Learning\_Guided\_Engineering\_Design\_from\_Topology\_Optimization\_to\_Advanced\_Modelling](https://www.researchgate.net/publication/389785241_Reinforcement_Learning_Guided_Engineering_Design_from_Topology_Optimization_to_Advanced_Modelling)

\[3\] [https://www.researchgate.net/publication/393164291\_Pseudo\_3D\_topology\_optimisation\_with\_reinforcement\_learning](https://www.researchgate.net/publication/393164291_Pseudo_3D_topology_optimisation_with_reinforcement_learning)

[![Giorgi Tskhondia](https://miro.medium.com/v2/resize:fill:96:96/1*7fA2QZN2SZZ-E7R_FvO4Fw.jpeg)](https://gigatskhondia.medium.com/?source=post_page---post_author_info--2bed7b1118e8---------------------------------------)[15 following](https://gigatskhondia.medium.com/following?source=post_page---post_author_info--2bed7b1118e8---------------------------------------)

An adept of reinforcement learning and finite element methods. Unlocking the fabric of reality @Gigala