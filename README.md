# coopsim
Disclaimer: This tool is a personal learning project. It can be used for educational purposes, but it is in no way attempting to simulate real-world situations.  

Developed with [Flask](https://flask.palletsprojects.com/en/3.0.x/)  
Currently not deployed anywhere.  

## Concept
The cooperation simulator randomises interactions between "cooperative" and "uncooperative" individuals in a generation. A fitness multiplier will be applied to both individuals of any interaction.  

A cooperative-cooperative interaction will apply the cooperative multiplier for both individuals.
An uncooperative-uncooperative interaction will apply the uncooperative multiplier for both individuals.
A cooperative-uncooperative interaction will apply the "disadvantageous" multiplier on the cooperative individual and the "advantageous" multiplier on the uncooperative individual.
The proportion of the sum of the fitness of all cooperative individuals on the sum of the fitness of all individuals defines the proportion of cooperative individuals in the next generation.

## Inputs
<b>Number of generations</b>: the total number of generations to simulate. Can be any positive integer number  

<b>Population size</b>: the number of individuals in each generation. Can be any positive integer number higher than 1.  

<b>Minimum interactions per individual</b>: each individual will be assigned <u>at least</u> this number of interactions. The randomisation algorithm iteratively checks for the number of connections
of each individual and, if the minimum is not met, will assign new connections to randomly picked individuals. It does not check for the connections already present in the receiving individual,
meaning that some individuals will end up having more interactions - particularly those that were picked out early in the iterative process. Can be any positive integer number smaller than the population size.  

<b>Initial proportion of uncooperative individuals</b>: self explanatory. Can be any number between 0 and 1.  

<b>Cooperative interaction fitness multiplier</b>: the fitness multiplier to be applied to each individual sharing a cooperative interaction. Can be any real positive number. Fractions are also accepted (e.g. 1/2 is the same as 0.5).  

<b>Uncooperative interaction fitness multiplier</b>: the fitness multiplier to be applied to each individual sharing an uncooperative interaction. Can be any real positive number. Fractions are also accepted (e.g. 1/2 is the same as 0.5).  

<b>Advantageous interaction fitness multiplier</b>: the fitness multiplier to be applied to uncooperative individuals when interacting with a cooperative individual  

<b>Disadvantageous interaction fitness multiplier</b>: the fitness multiplier to be applied to a cooperative individuals when interacting with an uncooperative individual  

<b>Seed</b>: a seed for repeatability purposes. Can be any positive integer number.  
