# MADDPG-Collaboration-and-Competition

Project: Collaboration and Competition

Introduction:

In this repository, we solve the environment “tennis” with a MADDPG-agent. The environment contains to tennis bats which have to hit the ball over the net. For every time they are able to hit the ball over the net they get a reward of +0.1. When the ball is hit out of bounds or a player lets the ball it the ground it gets a reward of -0.01. 
The environment is solved if the agent reaches an average reward of +0.5 over consecutive 100 attempts.

The state space has 24 dimensions and contains the speed of the bats, position of the ball etc. With these information the agent has to decide whether to go towards the net or away from it, and whether to go up or down. This is controlled by a number between -1 and +1. 

The agent can be trained by running the jupyter notebook “Teennis.ipynb”, while the information about the agent are stored in the file “ddpg_agent.py”. The neural networks are stored in the file “model.py”, while the weights of a trained networks are in the files “actor_model.pt” and “critic_model.pt”.

Needed Files:

To use the code you have to download "Tennis.ipynb" and the files “ddpg_agent.py” and “model.py”. Additionally, you have to download the environment "Tennis":

    Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip
    Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip
    Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip
    Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip

Additionally, one has to install requirements, which can be found in the file "requirements.txt". As an alternative one can directly install these packages using:

    pip3 install -r requirements.txt
