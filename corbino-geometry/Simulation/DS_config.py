"""
Jack Farrell, Dept. of Physics, University of Toronto, 2020

Settings for simulation to be run in annular_old.py.  This file should be in the
SAME directory as annular_old.py!
"""

k = 0.0005 # Time Step (units L / v_s)
h = 1 / 150. # Spatial Step Width (units

T = 150 # Total Length of Simulation (units L / v_s)

delta = 0.01# Entropy fix parameter (higher adds a bit more viscosity)

imageLog = True # If True, plots figures periodically so you can check on it
saveFigures = False
