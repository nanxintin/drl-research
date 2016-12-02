# One-step Q in Tensorflow + Keras + OpenAI's Gym


This is a Tensorflow + Keras implementation of asyncronous one-step Q learning as described in "Asynchronous Methods for Deep Reinforcement Learning".


It uses Keras to define the deep q network (see model.py), OpenAI's gym library to interact with the Atari Learning Environment (see atari_environment.py), and Tensorflow for optimization/execution (see one-step-Q.py).

## Requirements
* tensorflow, gym, gym's atari environment, skimage, Keras

## Usage
###Training

```
python one-step-Q.py --experiment spaceInvaders --game "SpaceInvaders-v0" --num_concurrent 8
```

###Visualizing training with tensorboard

```
tensorboard --logdir ./model/summaries/spaceInvaders
```

###Evaluation
To run a gym evaluation, turn the testing flag to True and hand in a current checkpoint file:
```
python one-step-Q.py --experiment spaceInvaders --testing True --checkpoint_path ./model/spaceInvaders.ckpt-269000 --num_eval_episodes 100
```
After completing the eval, we can upload our eval file to OpenAI's site as follows:

```python
import gym
gym.upload('./model/spaceInvaders/eval', api_key='YOUR_API_KEY')
```
Now we can find the eval at OpenAI.gym.


## Important notes
* This repo is based on coreylynch/async-rl project!
