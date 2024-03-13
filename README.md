## Description
This repository hosts a Python implementation of the Deep Q-Network (DQN) algorithm, a powerful method within the realm of Deep Reinforcement Learning (DRL). We apply DQN to solve the [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) environment provided by the Gymnasium library. The agent's objective is to navigate across a frozen lake, moving from the starting point to the goal while avoiding falling into any holes. The implementation aims to showcase the effectiveness of DQN in mastering this classic control problem while also serving as a reference for those interested in utilizing and practicing DQN.


## Introduction
The DQN algorithm is a value-based, model-free, and off-policy approach renowned for its capacity to learn optimal policies from high-dimensional input spaces. Originating from the efforts of researchers at DeepMind, DQN merges deep neural networks with traditional Q-learning to approximate the optimal state-action value function (Q function). The major pros and cons of the algorithm are as follows:



###### Advantages:
1. 	**Experience Replay Memory:** By utilizing exploration strategies like the epsilon-greedy policy and employing techniques such as experience replay, DQN significantly enhances sample efficiency and stabilizes the learning process for its main policy. This approach allows the algorithm to learn more effectively from past experiences and facilitates smoother convergence toward optimal policies.

###### Disadvantages:
1. 	**Hyperparameter Sensitivity:** DQN performance relies on tuning many hyperparameters, which makes it challenging to achieve optimal results in different environments.

2. 	**Training Instability:** During training, DQN may encounter instability, primarily originating from the dynamic nature of the target network. Furthermore, performance collapse can occur, presenting a scenario where DQN struggles to recover through learning, potentially hindering its training progress.




## Requirements
The code is implemented in Python 3.8.10 and has been tested on Windows 10 without encountering any issues. Below are the non-standard libraries and their corresponding versions used in writing the code:
<pre>
gymnasium==0.29.1
matplotlib==3.5.1
numpy==1.22.0
pygame==2.5.2
torch==2.0.1+cu118
</pre>

**Note:** This repository uses the latest version of Gymnasium for compatibility and optimization purposes. This code does not utilize any deprecated or old versions of the Gym library.



## Usage
The network weights for each map size (4x4 or 8x8) are stored in their corresponding directories: `./4x4/final_weights_1000.pth` or `./8x8/final_weights_3000.pth`. Therefore, there is no need to initiate training from the beginning when testing the code. Simply execute the code, and it will automatically load the weights, allowing for seamless testing with rendering enabled. Enjoy exploring the testing process!



## Showcase
You can view the training procedure through the following GIFs, demonstrating the learned process across episodes.

**Note:** The training was conducted with no randomness in the environment and without enabling the slippery mode (deterministic environment).


<div style="display: flex;">
  <img src="./Gifs/4x4 - EP 1.gif" width="31%" height=31% />
  <img src="./Gifs/4x4 - EP 500.gif" width="31%" height=31% />
  <img src="./Gifs/4x4 - EP 1000.gif" width="31%" height=31% />
</div>

<p align="center">
  <img src="./Gifs/8x8 - EP 1.gif" width="23%" height="23%" />
  <img src="./Gifs/8x8 - EP 900.gif" width="23%" height="23%" />
  <img src="./Gifs/8x8 - EP 1000.gif" width="23%" height="23%" />
  <img src="./Gifs/8x8 - EP 2500.gif" width="23%" height="23%" />
</p>



#### Results
The training outcomes for the 4x4 map size over 1000 episodes, and the 8x8 map size over 2500 episodes, are summarized below. This includes the raw rewards obtained and the Simple Moving Average of 50 (SMA 50) rewards:

<p align="left">
  <figure>
    <img src="./4x4/reward_plot.png" width="31%" height="31%" />
    <figcaption>4x4 map size</figcaption>
  </figure>
  <figure>
    <img src="./8x8/reward_plot.png" width="31%" height="31%" />
    <figcaption>8x8 map size</figcaption>
  </figure>
</p>


## Persian Tutorial Video
You can access the video tutorial (in persian) that explains the implementation of the DQN algorithm in Frozenlake environment from [here](https://youtu.be/lK4lfPGgGis).

I sincerely hope that this tutorial proves helpful to those of you who are in the process of learning. If you find this repository helpful in your learning journey, consider giving endorsement.