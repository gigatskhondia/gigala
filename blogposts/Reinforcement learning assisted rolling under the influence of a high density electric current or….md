---
title: "Reinforcement learning assisted rolling under the influence of a high density electric current or…"
source: "https://gigatskhondia.medium.com/reinforcement-learning-assisted-rolling-under-the-influence-of-a-high-density-electric-current-or-73ba12bc697b"
author:
  - "[[Giorgi Tskhondia]]"
published: 2026-02-18
created: 2026-04-19
description: "Reinforcement learning assisted rolling under the influence of a high density electric current or where AI meets the electroplasticity effect The era of RL assisted metal forming Co-authored with …"
tags:
  - "clippings"
---
The era of RL assisted metal forming

Co-authored with Claude

My PhD thesis was dedicated to an investigation of innovative techniques that lead to the improved plasticity and maintained strength of metal alloys, i.e. so called Electroplasticity Effect.

Electroplasticity Effect occurs during the process of plastic deformation when a high electric current of certain duration is applied to metal specimens. It leads to an improved plasticity and preserved strength — making it easier to plastically deform. It is a very useful technology in aerospace industry with the application e.g. to aluminium and titanium alloys \[1–2\].

Three competing mechanisms explain this:

- **Joule heating** — the current heats the material, softening it thermally
- **Electron-wind force** — moving electrons exert a drag force on dislocations, helping them move
- **De-pinning of dislocations from paramagnetic obstacles** — the electromagnetic field liberates dislocations that are stuck

My recent year passion, however, is topology optimization by reinforcement learning \[3–5\]. One can think of topology optimization by reinforcement learning as of an intelligent design. You can design machinery’s elements and components or you can design bridges and buildings.

An obvious extension of my research would be a combination of these domains — i.e. applying RL to the effect of electroplasticity.

For example, **RL + topology optimization + electroplasticity-informed manufacturing.**

Here how it would work. The rolling mill is now a robot (Fig 1); it automatically adapts to different alloys, pulse impacts, and rolling conditions, eliminating the need for manual adjustments. This can be very useful in **manufacturing**. How this is done: we have AI (specifically, RL), and we train it in a simulator, for instance, optimizing for maximum material plasticity (while preserving the strength). No data is needed. The simulator uses the finite element method. We create a controller (RL) on the computer and then transfer it to the real world (physical rolling mill).

![](https://miro.medium.com/v2/resize:fit:1256/format:webp/1*4fxa2EvO9eDPSl8Ugx0csQ.jpeg)

Figure 1. Rolling process of helicoids shaped rolls under the effect of a high-density electric current.

Electroplastic rolling is currently a highly manual, parameter-heavy process. An operator or engineer has to tune pulse frequency, current density, pulse duration, rolling speed, reduction ratio, temperature, and inter-pass timing — all simultaneously, all dependent on the specific alloy being processed. This is exactly the kind of high-dimensional, coupled, nonlinear control problem that RL was built for. Hence, RL assistance should allow to automate this process.

### How the System Would Actually Work

**The Simulator (Training Environment)**

The FEM simulator is the heart of it. You would build a coupled electro-thermo-mechanical FEM model. The RL agent observes the current state — roll force feedback, surface temperature, electrical measurements, pass number, alloy identity — and outputs actions: pulse parameters, rolling speed, reduction per pass, inter-pass intervals. The reward function is the critical design choice. Something like:

*Maximize elongation-to-failure (plasticity proxy) subject to: yield strength ≥ target, surface integrity constraints, energy consumption penalty.***Topology Optimization Layer**

Here is where it gets particularly interesting and where my TO with RL work enters. The rolling mill is not just a controller problem — the **geometry of the rolls themselves**, and potentially the geometry and distribution of electrode contacts delivering the current pulses, can be topology-optimized. You could use TO to design:

- Roll profile geometries that distribute stress and current more uniformly across complex cross-sections
- Electrode contact geometries that produce the most favorable current density distribution in the workpiece
- Die or guide geometries optimized jointly for mechanical and electrical boundary conditions

### Why This Is Industrially Compelling

The addressable problem is real and large. Aerospace manufacturers working with titanium alloys, nickel superalloys, and advanced aluminum alloys spend enormous effort on multi-pass rolling schedules. Getting those schedules wrong means scrapped material, which in titanium means very expensive scrap. An RL controller that automatically adapts to alloy-by-alloy variation and self-optimizes the pulse parameters could:

- Reduce forming force requirements by a large margin
- Enable forming of alloys currently considered too brittle for conventional rolling
- Eliminate the expert-knowledge bottleneck in schedule design
- Generalize across alloys without re-engineering the process from scratch

**Alloy-agnostic generalization** — is probably the most commercially valuable claim, because it is what eliminates the manual adjustment problem.

The era of RL assisted metal forming has just begun. Jump on!

To keep up with the project please visit [https://gigala.io/](https://gigala.io/)

\[1\] Simulating the effect of a high density electric current pulse on the stress field during plastic deformation. GA Tskhondiya, NN Beklemishev — International journal of material forming, 2012 — Springer

\[2\] Modeling the effect of multiple pulse treatment on deformation processing. GA Tskhondiya — Journal of Physics: Conference Series, 2009 — iopscience.iop.org

\[3\] [https://jngr5.com/index.php/journal-of-next-generation-resea/article/view/95](https://jngr5.com/index.php/journal-of-next-generation-resea/article/view/95)

\[4\] [https://www.sciencepublishinggroup.com/article/10.11648/j.ajasr.20261201.11](https://www.sciencepublishinggroup.com/article/10.11648/j.ajasr.20261201.11)

\[5\] [https://github.com/gigatskhondia/gigala](https://github.com/gigatskhondia/gigala)