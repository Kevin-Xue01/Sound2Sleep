# clas-python

## Documentation
Follow Docstring Conventions using numpy guidelines.

*  https://www.python.org/dev/peps/pep-0257/
*  https://numpydoc.readthedocs.io/en/latest/format.html

## Coding style
Use the Google Python Style Guide, except for Docstring

*  http://google.github.io/styleguide/pyguide.html

## Setup and initialization
**Note:** simpleaudio requires Microsoft Visual C++ 2005 build tools.


## Running


## Running modes
**CLAS**: (The full algorithm) Target specific phase and attempt to deliver stimulation exactly on target phase.  
**SHAM Muted**: (The full algorith, no auditory stim) Target specific phase and attempt to deliver stimulation exactly on target phase, but do not actually output any audio. Parallel port markers are are still outputted.
**SHAM Phase**: (previously "vary") Deliver stimulation on random phase. Rerandomize the target phase after each stimulation.
**SHAM Delay**: (previuosly "sham") Target specific phase, when stim is triggered, wait a random delay interval then deliver auditory stimuli.  