import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

def Buckingham_potential(a,b,c,r):
    """equation for the potential energy attraction between atoms in a solid / gas
    input: three variables a, b, c: 
    where a and b are part of the Pauli exclusion principle
    and c is the Van der Waals attraction equation.
    output: phi(r) which is potential in eV
    """
    pauli_exclusion = a*np.exp(-b*r)
    vanderWaals = -c/r**6
    return pauli_exclusion + vanderWaals

if __name__ == "__main__":
       
    angstroms = np.linspace(0.8,3,50) #range of values to be computed by Buckingham_potential
    phi = Buckingham_potential(1,1,1,r = angstroms) 
    
    fun = lambda r: np.exp(-1*r[0])-(1/r[0]**6) #defines function for minimize scipy code
    maxima = minimize(fun,x0 = (1.2,1)) #Does not return the correct minimum (even though here we want max)
    print('\n')
    print('For this function, the maxima is at:',maxima.fun)
    
    #Code for plotting function
    plt.plot(angstroms,phi,color = 'crimson')
    plt.ylabel('eV')
    plt.xlabel('Angstroms')
    plt.title('Buckingham Potential')
    plt.show()