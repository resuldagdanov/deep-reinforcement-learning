# DRL Homework 1

### Topics
- Markov Decision Process
- Dynamic Programming
- Temporal Difference Learning and Monte Carlo
- Approximate methods in TD

### Structure

Follow the ipython notebooks with the order given below:

- MDP
- dynamic_programming
- learning
  
In each notebook, you will have questions. There are two types of questions, written and implementation based, both of them are marked as **Question x)**. You need to fill the cell below for the written questions or implement the part of the code stated in the implementation questions.

You can run the test script for dynamic programming code using: (**Optional**)

```
python -m unittest test_dp.py
```

### Installation

To start your homework, you need to install requirements. We recommend that you use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment for this homework.

```
conda create -n dlhw1 python=3.7
conda activate dlhw1
```
You can install requirements with the following command in the homework directory:

```
pip install -r requirements.txt
```
Then you need to install the homework package. You can install the package with the following command:

```
pip install -e .
```

This command will install the homework package in development mode so that the installation location will be the current directory.

### Docker

You can also use docker to work on your homework instead of following installation steps. Simply, build a docker image in the homework directory using the following command:

```
docker build -t rlhw1 .
```

You may need to install docker first if you don't have it already.

After building a container we need to mount the homework directory at your local computer to the container we want to run. Note that the container will install necessary python packages inbuilt.

You can run the container using the command below as long as your current directory is the homework directory:

```
sudo docker run -it --rm -p 8889:8889 -v $PWD:/hw1 rlhw1
```

This way you can connect the container at ```localhost:8889``` in your browser. Note that, although we are using docker, changes are made in your local directory since we mounted it. 

You can also use it interactively by simply running:

``` 
sudo docker run -it --rm -p 8889:8889 -v $PWD:/hw1 rlhw1 /bin/bash
```

### Submission

Any large-sized file (images, binaries, model parameters, etc) should be ignored while submitting the homework. Submissions are done via Ninova until the submission deadline. Submit the homework directory as a zip file when you complete the homework.

### Evaluation

- MDP (20%)
- DP (30%)
- Learning (50%)

### Related Readings

> Reinforcement Learning:
An Introduction
second edition
Richard S. Sutton and Andrew G. Barto

- MDP: chapter 3
- Dynamic programming: chapter 4
- Monte Carlo: chapter 5
- Tabular TD methods: chapter 6 & 7
- approximate TD methods: chapter 9

### Contact

TA: Tolga Ok
okt@itu.edu.tr
