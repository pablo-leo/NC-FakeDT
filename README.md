# Installation
In order to execute the code, it is necessary to previously install some packages. The python version should be 3.6.4

## Dependencies
  - numpy 1.14.5
  - sklearn 0.19.1
  - scipy 1.1.0
  - pandas 0.23.1
  - nltk 3.3
  - keras 2.2.0
  - tensorflow 1.8.0
  - gensim 3.4.0

Alternatively you can execute the 'install_dependecies.sh' file which installs all the required packages in the correct version, but make sure all the packages are previously uninstalled in the machine.

# Execution
To execute the code, just open a terminal inside the root directory of the project and execute the python file 'fake_trump.py', including (if wanted), some of the following flags.
 - --pop     (population size)
 - --it      (number of iterations)
 - --cross   (crossover rate)
 - --mut     (mutation rate)
 - --top     (topics, separated by spaces)
 
Execution example:
$ python fake_trump.py --pop 200 --it 10 --cross 0.3 --mut 0.1 --top America news
