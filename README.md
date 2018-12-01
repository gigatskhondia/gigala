# Engineering Design by Reinforcement Learning and Finite Element Analysis
Models in repository aim to optimize design of a structure via combining finite element analysis and reinforcement learning. The finite element model represents an environment to which an agent applies actions and from which it gets observations and rewards. Agent uses neural network policy gradient algorithm. The algorithm optimizes the parameters of a policy by following the gradients toward higher rewards. The end result of the modeling is an optimal (in terms of displacements) design of a structure.

## 1. Simple finite element analysis optimized by reinforcement learning 

  A bar is subjected to axial tensile force. The model optimizes by altering cross-sectional area and material of the bar in order to keep the maximum displacement in the 1D structure within acceptable limit.  An action, in this case, is altering parameters of cross-sectional area and Young's modulus in a certain way.  If an action leads to alleviation of displacements, measured by FEA, the agent gets a reward.


## 2. Design of pin-jointed frame structure by reinforcement learning
  
  2D truss is subjected to a vertical force at its tip node. The model optimizes positions of truss joints in order to minimize displacements in the structure’s tip node. The agent gets a reward if “tip” displacement is reduced due to change in position of truss joints. The model has serious memory leaks due to usage of external .exe file in the code. For 2D truss structures use "bridge" model V.03. 


## 3. Design of a bridge by reinforcement learning

  The model optimizes positions of joints in order to minimize vertical displacement in middle node of the bridge like 2D truss structure. The agent gets a reward if the middle node displacement is reduced due to change in position of truss joints. At the end of notebook one can see the final design of the “bridge” produced by AI. V.03 of the notebook is stable and does NOT have any memory leaks.


## 4. Spool design by plane frame element and reinforcement learning

  This model optimizes geometry in order to minimize maximum displacement in a “spool”. The agent gets a reward if the maximum nodal displacement is reduced due to change in the spool geometry. The FE model uses a plane frame element.
