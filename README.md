# Custom-ML-Agents-Trainer
# Objective:
Develop a framework for RL implementation into the Unity Game engine with the driving and navigation task.

# Initial goals for RL Unity interface: 
Well documented, python compatibility, continued support. 

# ML Agents:
Through ML Agents we are able to utilize a Python API to create our own models that interface with the objects of a Unity game. Separately, pre-made models like PPO and SAC are usable to train the agent in the environment. These models can serve two purposes. One advantage of having pre-made models is for the obvious reason that they don't have to be custom made, saving time. Another advantage  is that they can help confirm whether the environment is set up correctly without having to create a model to confirm. These models run through the Unity editor unlike custom models, which require the game to be exported. Performance plots are viewable in tensorboard. When running a premade model, modifying the .yaml file allows for some refinement of parameters like learning rate, batch size, buffer size, max steps,  etc. This provides some flexibility when using pre-made models. However these premade models include references to many other packages and files, making them difficult to understand or diagnose fully which is why a custom model is advantageous. There are a few functions that are necessary from the Ml agents package to implement into the Python code when creatmg a custom training algorithm. Those functions are listed below:

https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Python-LLAPI.md

Functions of ml agents functions in python:
# Read and store the Behavior Name of the Environment
behavior_name = list(env.behavior_specs)[0]

 # Read and store the Behavior Specs of the Environment
 spec = env.behavior_specs[behavior_name]

# Get the Decision Steps and Terminal Steps of the Agents
decision_steps, terminal_steps = env.get_steps(behavior_name)

# Set the actions in the environment
# Unity Environments expect ActionTuple instances.
      action_tuple = ActionTuple()
      action_tuple.add_discrete(actions)
      env.set_actions(behavior_name, action_tuple)

# Perform a step in the simulation
      env.step()

# Reset the environment
    env.reset()

# How to Get ML Agents Working:
https://unity-technologies.github.io/ml-agents/Getting-Started/

Open/Create a Unity project
Install ml agents package: Window>Package Manager> ML Agents
Write A c# script to define the Agents bahavior anmd reward scheme in Unity.
Agent Object inspector: 
Add a Behavior Parameters Component (defines how an agent makes decisions)
Define Space Size (vector of floating point numbers) and Continuous/Discrete actions 


Pre Built models:
In the terminal, navigate to the directory of the Unity project
> pip3 install mlagents
> pip3 install torch torchvision torchaudio
> pip3 install protobuf==3.20.3

Running the training sequence:
	> mlagents-learn --run-id=Test1

To see training results, in terminal type: 
> tensorboard --logdir results --port 6006
	Copy the link it provides to the tensorboard site

Custom Models:
Navigate to File>Build Settings>Build  (make sure development build is checked)
Place the file in the same location as the python trainer code.
…
Include the file name in the python trainer code.
