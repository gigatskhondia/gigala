# Reinforcement_Learning_and_FEA
Combining reinforcement learning and finite element analysis

## 1. Simple finite element analysis optimized by reinforcement learning: 

  General idea of the model is to optimize design of a structure or component via combination of FEA and reinforcement learning. In     particular, you have a bar that is subjected to axial tensile force. You optimize for cross-sectional area and material for the bar in order to make the maximum displacement stay within acceptable bounds and not exceed critical value in any point. 

  In reinforcement learning terms, the finite element model represents an environment to which you apply actions. An action, in this case, is altering parameters of cross-sectional area and Young's modulus in a certain way. If an action leads to alleviation of displacements, measured by FEA, the RL part of the model gets a reward.

  RL model uses neural network policy gradient algorithm. The algorithm optimizes the parameters of a policy by following the gradients toward higher rewards. The end result of the modeling is an optimal cross-sectional area and Young's modulus that preserve displacements within acceptable limits.
