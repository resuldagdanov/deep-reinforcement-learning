# DRL Homework 2

### Topics
- DQN
- Rainbow
    - Double Q-learning
    - Prioritized Replay
    - Dueling Network
    - Multi-step Learning
    - Distributional RL
    - Noisy Nets

### Structure

Follow the "hw.ipynb" notebook for instructions:


You can run the **optional** test script that covers replay buffers from the "test" directory using:

```
python -m unittest test_replaybuffer.py
```

### Installation

To start your homework, you need to install requirements. We recommend that you use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment for this homework.

```
conda create -n rlhw2 python=3.7
conda activate rlhw2
```

If you are going to use GPU, install [Pytorch](https://pytorch.org/get-started/locally/) using the link and remove it from requirements.

You can install the requirements with the following commands in the homework directory:

```
conda install -c conda-forge swig
conda install nodejs
pip install -r requirements.txt
python -m ipykernel install --user --name=rlhw2
```
Then you need to install the homework package. You can install the package with the following command: (Make sure that you are at the homework directory.)

```
pip install -e .
```

This command will install the homework package in development mode so that the installation location will be the current directory.


### Docker

You can also use docker to work on your homework. Build a docker image from the homework directory using the following command:

```
docker build -t rlhw2 .
```

You may need to install docker first if you don't have it already.

After building a container we need to mount the homework directory at your local computer to the container we want to run. Note that the container will install necessary python packages in build.

You can run the container using the command below as long as your current directory is the homework directory:

```
sudo docker run -it --rm -p 8889:8889 -v $PWD:/hw2 rlhw2
```

This way you can connect the container at ```localhost:8889``` in your browser. The container will automatically run Jupyter-Notebook. Note that, although we are using docker, changes are made in your local directory since we mounted it.

You can also use it interactively by simply running:

```
sudo docker run -it --rm -p 8889:8889 -v $PWD:/hw2 rlhw2 /bin/bash
```

> Note: Running docker with cuda requires additional steps!

### Submission

You need to submit this repository after you filled it (and additional files that you used if there happens to be any). You also need to fill "logs" directory with the results of your experiments as instructed in the Jupyter notebook (progress.csv files that you used in your plots). Please only include the valid and necessary log files in your final submission. Submissions are done via Ninova until the submission deadline. For the atari model parameters, you should put a google drive link in the Jupyter notebook.

### Evaluation


- Implementations 50%
    - DQN (25%)
    - RAINBOW (75%)
        - Prioritized Replay 20%
        - Distributional RL 20%
        - Noisy Nets 15%
        - Multi-step learning 10%
        - Dueling Networks 5%
        - Double Q-learning 5%
- Experiments 50%
    - LunarLander (75%)
    - Pong (25%)



### Related Readings

- [DQN](https://www.nature.com/articles/nature14236)
- [Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf)
- [Prioritized Replay](https://arxiv.org/pdf/1511.05952.pdf)
- [Dueling Network](https://arxiv.org/pdf/1511.06581.pdf)
- Multi-step Learning - Richard S. Sutton and Andrew G. Barto Chapter 7
- [Distributional RL](https://arxiv.org/pdf/1707.06887.pdf)
- [Noisy Nets](https://arxiv.org/pdf/1706.10295.pdf)
- [Rainbow](https://arxiv.org/pdf/1710.02298.pdf)

### Contact
TA: Tolga Ok
okt@itu.edu.tr
