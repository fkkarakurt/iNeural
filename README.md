<p align="center">
  <img width="250" src="https://raw.githubusercontent.com/fkkarakurt/iNeural/main/iNeural.png">
</p>

# iNeural

A library for creating Artificial Neural Networks, for use in Machine Learning and Deep Learning algorithms.

## What is a Neural Network?

Work on artificial neural networks, commonly referred to as “neural networks,” has been motivated right from its inception by the recognition that the human brain computes in an entirely different way from the conventional digital computer. The brain is a highly complex, nonlinear, and parallel computer (information-processing system). It has the capability to organize its structural constituents, known as neurons, so as to perform certain computations (e.g., pattern recognition, perception, and motor control) many times faster than the fastest digital computer in existence today. Consider, for example, human vision, which is an information-processing task. It is the function of the visual system to provide a representation of the environment around us and, more important, to supply the information we need to interact with the environment. To be specific, the brain routinely accomplishes perceptual recognition tasks (e.g., recognizing a familiar face embedded in an unfamiliar scene) in approximately 100–200 ms, whereas tasks of much lesser complexity take a great deal longer on a powerful computer.

For another example, consider the sonar of a bat. Sonar is an active echolocation system. In addition to providing information about how far away a target (e.g., a flying insect) is, bat sonar conveys information about the relative velocity of the target, the size of the target, the size of various features of the target, and the azimuth and elevation of the target. The complex neural computations needed to extract all this information from the target echo occur within a brain the size of a plum. Indeed, an echolocating bat can pursue and capture its target with a facility and success rate that would be the envy of a radar or sonar engineer.

How, then, does a human brain or the brain of a bat do it? At birth, a brain already has considerable structure and the ability to build up its own rules of behavior through what we usually refer to as “experience.” Indeed, experience is built up over time, with much of the development (i.e., hardwiring) of the human brain taking place during the first two years from birth, but the development continues well beyond that stage.

A “developing” nervous system is synonymous with a plastic brain: Plasticity permits the developing nervous system to adapt to its surrounding environment. Just as plasticity appears to be essential to the functioning of neurons as information-processing units in the human brain, so it is with neural networks made up of artificial neurons. In its most general form, a neural network is a machine that is designed to model the way in which the brain performs a particular task or function of interest; the network is usually implemented by using electronic components or is simulated in software on a digital computer. In this book, we focus on an important class of neural networks that perform useful computations through a process of learning. To achieve good performance, neural networks employ a massive interconnection of simple computing cells referred to as “neurons” or “processing units.” We may thus offer the following definition of a neural network viewed as an adaptive machine.

_A neural network is a massively parallel distributed processor made up of simple processing units that has a natural propensity for storing experiential knowledge and making it available for use. It resembles the brain in two respects:_

**1. Knowledge is acquired by the network from its environment through a learning process.**

**2. Interneuron connection strengths, known as synaptic weights, are used to store the acquired knowledge.**

The procedure used to perform the learning process is called a learning algorithm, the function of which is to modify the synaptic weights of the network in an orderly fashion to attain a desired design objective.

The modification of synaptic weights provides the traditional method for the design of neural networks. Such an approach is the closest to linear adaptive filter theory, which is already well established and successfully applied in many diverse fields. However, it is also possible for a neural network to modify its own topology, which is motivated by the fact that neurons in the human brain can die and new synaptic connections can grow

<p  align="right"><em>Neural Networks and Learning Machines, 3rd Edition</em></p>
<p align="right"><b>Simon Haykin</b></p>

## What Does iNeural Do?

iNeural is an open source library for artificial neural networks. One of the best things about iNeural is that it has very few dependencies. It only needs a few external libraries and tools. Other than that, everything he needs is already coded. _It is not prepared for GPUs to run at full performance._ But it is a library that can work great with low system requirements. To give an example, when the project is completely finished, you will be able to run it with high performance on robotic platforms.

So who is **iNeural** suitable for?

- For those who want to use open source neural network in problem solving,

- For those who want to integrate new technologies into their projects,

- For students who want to understand the tricks needed for neural networks,

- Researchers, Machine Learning and Deep Learning Enthusiasts.

## Why Developing iNeural?

It is developed by taking inspiration from libraries such as iNeural, [FANN](https://github.com/libfann/fann), [pylearn2](https://github.com/lisa-lab/pylearn2), [EBLearn](http://eblearn.sourceforge.net/), [Torch7](http://torch.ch/). Written mostly in C++, iNeural also leverages the power of Python. The biggest reason for its development is that it needs very few dependencies. For this reason, it is expected to be suitable for working in systems with limited system requirements.

## Who Is iNeural Developing By?

iNeural is being developed by [Fatih Küçükkarakurt](https://github.com/fkkarakurt). All expenses and programming of the project are done by him.

## How Can I Contribute to the Project?

Since iNeural is a very new project, there is a lot of things you can contribute to. For this, you can clone the project after forking.

For this run the following line:

`git clone https://github.com/fkkarakurt/iNeural.git`

I like to use `CMake` as a compiler tool. So I will use `CMake` in this project as well. However, since it is an unfinished project, I left the editing for later. For this, cloning the project is only for you to examine the codes.

- Before you open a ticket or send a pull request, [search](https://github.com/fkkarakurt/iNeural/issues) for previous discussions about the same feature or issue. Add to the earlier ticket if you find one.

- Use the same coding style as the rest of the [codebase](https://github.com/fkkarakurt/iNeural/tree/master/src). If you are using Visual Studio Code, I request you to use [Prettier](https://prettier.io/) and [C/C++](https://code.visualstudio.com/docs/languages/cpp) plugins.

## How Long Does the Project Take to Complete?

I make additions to the project every day. In fact, most of the libraries I have programmed are readily available. Just doing the final tests to `push` here. I want to make sure everything really works. There are parts that I missed, and I don't want to overwhelm you with these problems.

_I am preparing a very detailed documentation describing what the project really does and how to use it. I hope I `push` as soon as possible._

## Technologies in Use

- CMake v3.21.0-rc2
- GNU C++ Compiler (g++)
- [Eigen 3](https://eigen.tuxfamily.org/)
- [ALGLIB](https://www.alglib.net/)
- Python 3
- C++

## Test Data and Datasets

- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [IRIS](https://archive.ics.uci.edu/ml/datasets/iris)
- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [OCTOPUSARM](https://www.cs.mcgill.ca/~dprecup/workshops/ICML06/octopus.html)
- [THE POLE BALANCING PROBLEM](<https://researchbank.swinburne.edu.au/file/62a8df69-4a2c-407f-8040-5ac533fc2787/1/PDF%20(12%20pages).pdf>)
- [SARCOS](http://www.gaussianprocess.org/gpml/data/)

---

<p align="center">
<a href="https://linkedin.com/in/fkkarakurt" target="_blank"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"></img></a><a href="https://instagram.com/fkkarakurt" target="_blank"><img src="https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white"></img></a><a href="https://twitter.com/fkkarakurt" target="_blank"><img src="https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white"></img></a><a href="https://github.com/fkkarakurt" target="_blank"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white"></img></a><a href="https://www.hackerrank.com/fatihkkarakurt11" target="_blank"><img src="https://img.shields.io/badge/-Hackerrank-2EC866?style=for-the-badge&logo=HackerRank&logoColor=white"></img></a></p>

---

<p align="center"><img src="https://github-readme-stats.vercel.app/api?username=fkkarakurt"></img></p><p align="center"><img src="https://github-readme-stats.vercel.app/api/top-langs/?username=fkkarakurt"></img></p>
