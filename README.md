# Engineering Design by Artificial Intelligence

Are you interested in the new ways of engineering design? This repository is an attempt to apply artificial intelligence algorithms for the purpose of engineering design of electrical and structural elements and components. I combine numerical simulation like finite element analysis with artificial intelligence like reinforcement learning and genetic algorithms to produce optimal designs. In fact, to the best of my knowledge, I was pioneering the combination of RL and FEA for the purpose of structural engineering design since 2018. Recently, my work has been focused on the analysis of dynamics of offshore structures, electrical circuits and topology optimization. I am constantly exploring different ways that AI can be applied to science and engineering. With my diverse interests, I am using this repository as a testbed for my ideas to create the industry standard software for artificial intelligence aided design. I hope that my work can inspire you to explore the different ways that AI can be applied to your field.
 
 
Please check my [blog](https://gigatskhondia.medium.com/) and [manuals](https://gigatskhondia.github.io/gigala/) for the specifics of the models and algorithms I use:

* [Engineering Design by Reinforcement Learning and Finite Element Methods.](https://gigatskhondia.medium.com/engineering-design-by-reinforcement-learning-and-finite-element-methods-82eb57796424)
* [Topology optimization with reinforcement learning.](https://gigatskhondia.medium.com/topology-optimization-with-reinforcement-learning-d69688ba4fb4)
* [Some philosophical aspects of artificial intelligence.](https://gigatskhondia.medium.com/some-philosophical-aspects-of-artificial-intelligence-0a0f0bdb6312)
* [Engineering Design by Genetic Algorithms and Finite Element Methods.](https://gigatskhondia.medium.com/engineering-design-by-genetic-algorithms-and-finite-element-methods-5077ebadd16e)
* [On artificial intelligence aided engineering design.](https://gigatskhondia.medium.com/on-artificial-intelligence-aided-engineering-design-a6cf6f76b3d9)
* [Some ideas on using reinforcement learning in marine construction and sustainable energy development.](https://gigatskhondia.medium.com/using-reinforcement-learning-in-marine-construction-and-sustainable-energy-development-b5f301fb2397)
* [Modelling offshore pipelaying dynamics.](https://medium.com/@gigatskhondia/modelling-pipelay-dynamics-with-second-order-ordinary-differential-equation-using-python-4d6fc24055b)
* [Modelling offshore pipelaying dynamics - Part 2.](https://gigatskhondia.medium.com/modelling-offshore-pipelaying-dynamics-part-2-in-6-dof-a360965a7a89)
* [Pipelay vessel design optimisation using genetic algorithms.](https://medium.com/@gigatskhondia/pipelay-vessel-design-optimisation-using-genetic-algorithms-506aa04212f1)
* [Python and Finite Element Methods: A Match Made in Heaven?](https://gigatskhondia.medium.com/python-and-finite-element-methods-a-match-made-in-heaven-ee2ed7ca14ee)


Design of bionic partition (GA on the left, and RL on the right): 

<img width="342" alt="Screenshot 2023-06-29 at 15 56 49" src="https://github.com/gigatskhondia/gigala/assets/31343916/54689109-65ec-4b4c-87ae-1fe11dba031c"><img width="371" alt="Screenshot 2023-07-16 at 22 45 56" src="https://github.com/gigatskhondia/gigala/assets/31343916/4d5954dc-5e80-4b8e-8d02-1ab5757281df">

Topology optimization by reinforcement learning:

<img width="165" alt="Screenshot 2023-07-31 at 22 44 45" src="https://github.com/gigatskhondia/gigala/assets/31343916/bde9577c-0647-4c29-82b1-8f402deff7b0"><img width="277" alt="Screenshot 2023-08-02 at 12 39 14" src="https://github.com/gigatskhondia/gigala/assets/31343916/e97365d9-71cc-4c15-a790-4cb04037c163"><img width="241" alt="Screenshot 2023-08-05 at 15 40 24" src="https://github.com/gigatskhondia/gigala/assets/31343916/48246d51-37bc-4858-8490-904441c1063b">


#### Latest ideas: ####
On how to apply recent developments in Generative AI to my solution: you can feed an image to chatGPT and ask it questions about this image. For example, you can ask 'do you like this image?'. In case of designing a structure like a bridge, you can ask 'do you like this design of a bridge?'. Based on chatGPT's answer you can calculate a sentiment score. You can feed this sentiment score to your RL agent as additional reward (making it huge). You do not have to ask chatGPT every MDP iteration step, just once in a while. After you get your final geometry, you can make it look pretty or 3D by enhancing it with Stable Diffusion.

<img width="568" alt="Screenshot 2023-05-16 at 11 37 50" src="https://github.com/gigatskhondia/gigala/assets/31343916/ef983d4e-e8f6-456e-80b9-d2fa95aba1d8">

#### TODO: ####
  
*  Design MEMS
   *  Force inverting mechanism
   *  Use SfePy for FEM ?
   *  Use action as part of observation?
   *  Do not need to remesh, just set stiffness to a small number for absent elements 
   *  Use GNN or CNN for features ?
   *  Measure RL trianing progress
   *  Experiment with different objective functions e.g., compliance, stress based, etc.
   *  Test generalizability
*  Engine pylon
   *  Use Elmer FEM or SfePy for the engine
   *  Use python for topology design zone
   *  Hierarchical RL
   *  Experiment with PINN instead of FEM 
*  Validate robotic and PDE pipe models

To keep up to date with the project please check [Gigala](https://gigala.io/) page.

#### If you like my project and want to support it, please consider doing any of the following: ####
* Star this project
* [Sponsor](https://www.paypal.me/gigatskhondia) this project 
* [Contact](https://gigala.io/) me if you are interested in my research or want to adopt my solutions for your project
