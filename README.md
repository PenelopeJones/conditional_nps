"""
Implementation of Conditional Neural Processes in PyTorch. 

Investigation into methods for training a CNP on a single dataset (i.e a single instance of a function) 
using bootstrap-like techniques, as opposed to training on several functions assumed to be drawn from the same
stochastic process. 

Based on the work carried out in this paper: 
Conditional Neural Processes: Garnelo M, Rosenbaum D, Maddison CJ, Ramalho T, Saxton D, Shanahan M,
Teh YW, Rezende DJ, Eslami SM. Conditional Neural Processes. In International Conference on Machine
Learning 2018.

Requirements: 
PyTorch, numpy, sklearn
"""
