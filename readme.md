# Carpole 

## Introduction 

This is a side project which I implement the deep Q network(DQN) to solve the cartpole problem. In this project I use the environment built by OpenAI gym. 

The goal of the agent is to keep the pole on the cart as long as possible, the cart can move left or right to balance it. The agent will get a +1 reward for every step taken.

## How to run this project
You need to install the packages such as [torch](https://pytorch.org/get-started/locally/) from pytorch and [gym](https://github.com/openai/gym) from OpenAI

### Tsrain
You can run the program by

```
$ python main.py
```
### Display
which will start training the agent, and you can see the result from **rewards.png**.

To visualize the game, run 
```
$ python display.py 
```
The `display` will run the saved agent 10 times.

## Result
According to the documentation, the maximum reward an agent get can get is +500 per episode. 

![Result]('rewards.png')

Although the noise during the training process is huge, the average reward is growing steadily which shows the agent is learning to do its job quite well.



## Reference
1. [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf?source=post_page)
