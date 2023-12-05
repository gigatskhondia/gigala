### TODO: work in progress, the environment not optimally solved yet!


# Hierarchical reinforcement learning for topology optimization

This is an implementation of Hierarchical-Actor-Critic-HAC algorithm for topology optimization of cantilever beam. 
The codebase was taken from [Nikhil Barhate](https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch)  and adjusted for the topology optimization task at hand.


# Hierarchical-Actor-Critic-HAC-PyTorch

Hierarchical Actor Critic (HAC) algorithm described in the paper, [Learning Multi-Level Hierarchies with Hindsight](https://arxiv.org/abs/1712.00948) (ICLR 2019). The algorithm learns to reach a goal state by dividing the task into short horizon intermediate goals (subgoals). 


## Usage
- All the hyperparameters are found by `hpo.py`.
- To train a new network run `train.py`
- To test a preTrained network run `test.py`
- For a detailed explanation of offsets and bounds, refer to [issue #2](https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch/issues/2)


## Implementation Details

- The code is implemented as described in the appendix section of the paper and the Official repository, i.e. without target networks and with bounded Q-values.
- Topology optimization model is taken from [A Tutorial on Structural Optimization](https://www.researchgate.net/publication/360698153_A_Tutorial_on_Structural_Optimization) 
- Implementation tutorial: [Using Hierarchical Reinforcement Learning for Fast Topology Optimisation](https://gigatskhondia.medium.com/using-hierarchical-reinforcement-learning-for-fast-topology-optimisation-85aa0c07fb7f)


## Citing

- [How to cite](https://github.com/gigatskhondia/gigala/blob/master/CITATION.cff)

## Requirements

- Python 
- PyTorch
- OpenAI gym
