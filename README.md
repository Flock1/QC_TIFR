## Equal Superposition using Reinforcement Learning

This repo consists of a reinforcement learning implementation that has learned to convert any random quantum state to equal superposition. 

## USAGE

`code_clean.ipynb` consists of the clean implementation of the training and testing process. This implementation uses vanilla policy gradient to train the model. Vanilla policy gradient is an efficient on-policy training technique that has worked for a lot of Atari games. This has been developed using Keras library. There are two trained models, `model_July_7.h5` and `model_June_23.h5`. `model_July_7.h5` seems to be doing a better job than the other one. 

The second implementation is `torch.ipynb` that uses PyTorch and was trained using the Proximal Policy Optimization (PPO), which was developed by OpenAI.

## REQUIREMENTS

You will need `Qiskit` to run both the codes. For `code_clean.ipynb`, Keras with Tensorflow backend (`pip install Keras`) is needed along with numpy, matplotlib, math and random. 

For `torch.ipynb`, you will need pytorch library (`pip install torch torchvision`) along with the packages mentioned above. 