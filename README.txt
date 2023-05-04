This is a proof-of-concept RL and synthesizer combo that trains an RL model to obey arbitrary constraints that can be synthesized, entered, or randomized once training
is finished. 


REQUIREMENTS:
python3
z3
torch
pickle
argparse




The game is simple: An agent begins at 0,0 on the XY plane and receives the angle of the nearest reward, along with two additional parameters: a minimum
and maximum angle. The goal is to collect as many rewards as possible in 200 steps. 


The novel concept here is that the model is not trained on a specific safety policy; during training, the limits are randomized to be
in the interval (0, 4 * pi), with the upper limit never being more than 2 * pi higher than the lower limit. This value represents the angle that the 
the agent will move in. The current synthesized policy, which can be changed by editing the synthesize_constraints function, synthesizes a constraint that restricts turning
to up to pi/2 in either direction from the last angle that the agent moved in. The exciting thing is not that the agent can learn this specific policy- but that
we can implement an arbitrary safety synthesizer that can guarantee that the agent will satisfy those constraints(the guaranteee can be relaxed, see the options section)



HOW TO USE:

Training mode:
To use in training mode, use the following command:
python3 finalprojv2.py <number of training epochs> <number of epochs without synthesis> <neural network size> <-v, -vv, -vvv>

I would recommend a neural network size of at least 128, preferablly 256. I don't think more is necessary. For proper training, at least 3000 epochs is
recommended. I would not do more than a few(if at all) without synthesis; the second you start training it on an actual policy, it will start
learning that specific safety policy instead of focusing on the constraints. If you want more information during training, add -v,-vv, or -vvv.

If I ran 

python3 finalprojv2.py 5 1 10
It would have 4 epochs without synthesis, one with, and the actor neural network would have two hidden layers with ten nodes each (and the critic would have 2 with 7 each).


At the end of training, it will save the model to disk.

Testing Mode:
To use in testing mode, use the following command:
python3 finalprojv2.py <number of training epochs> <number of epochs without synthesis> <neural network size> -nt <-v, -vv, -vvv>


This will run the model from disk, but it will not learn from it, and will simply display the information on the screen.

Manual mode:
To use in manual testing mode, use the following command:
python3 finalprojv2.py <number of training epochs> <number of epochs without synthesis> <neural network size> -mi

This allows the user to enter limits for the model and see how it responds.









