# Try with Crank Nicholson perhaps. 
from crank_nicolson import *

initial = (lambda x: 2*np.pi*x - np.sin(2*np.pi*x))
# Need to implement the Neumann conditions in the ends also!
