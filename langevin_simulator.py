'''
Implements the Langevin equation in 1D. The particle starts at 0 with a negative velocity sampled from the Maxwell-Boltzmann Distribution
'''
from boltzmann_sampler import sample_maxwell_boltzmann_distribution
import numpy as np
import constants

class LangevinSimulation():
    '''
A simple example class to simulate Langevin motion.

This class models the dynamics of a particle using the Langevin equation, considering thermal fluctuations, friction, and external forces. It computes the particle's position and velocity over time.

Attributes:
    trajectory (list): List of positions recorded during the simulation.
    thermal_fluctuation_deviation (float): Standard deviation for thermal fluctuations.
    boltzmann_deviation (float): Standard deviation for Maxwell-Boltzmann distribution.
    root_a_over_b (float): Term for the force equation.
    a (float): Constant 'a' for the force equation.
    friction (float): Friction term (drag coefficient).
    x (float): Current position of the particle.
    velocity (float): Current velocity of the particle.
    initial_velocity (float): Initial velocity of the particle.
    force (float): Current force acting on the particle.
    m (float): Mass of the particle.
    
Methods:
    __init__(self):
        Initializes the simulation with initial variables.
        
    _set_initial_variables(self):
        Sets up the initial variables like thermal fluctuations, velocity, force, etc.
        
    force_at_x(self):
        Calculates and returns the force at the current position.
        
    _sample_thermal_fluctuation(self):
        Samples a thermal fluctuation based on the thermal fluctuation deviation.
        
    calculate_langevin_forces(self):
        Calculates the forces on the particle at each timestep based on Langevin dynamics.
        
    propagate_velocity(self, timestep:float=constants.dt):
        Updates the particle's velocity using the Langevin equation.
        
    propagage_position(self, timestep:float=constants.dt):
        Updates the particle's position based on the velocity and timestep.
        
    simulate_langevin_motion(self, timestep:float=constants.dt, numsteps:int=constants.steps):
        Runs the Langevin simulation over a number of timesteps, updating the position, velocity, and forces.
    '''
    def __init__(self):
        self._set_initial_variables()

    def _set_initial_variables(self):
        '''
Sets up the initial variables for the simulation, including thermal fluctuations, 
velocity, friction, and force constants. Handles missing constants by using default values.
        '''
        self.trajectory = []

         # Set up std deviation for thermal distribution
        try:
            second_moment = constants.friction*(constants.T*constants.k_b)
        except:
            second_moment = 0
            print('Cannot set a correct second moment of the correlation distribution. Defaulting to 0.')
        self.thermal_fluctuation_deviation = np.sqrt(second_moment)

        # Set up std deviation for maxwell-boltzmann distribution
        try:
            boltzmann_variance = np.sqrt(constants.k_b*constants.T/constants.m)
        except:
            boltzmann_variance = 1
            print('Cannot set up a proper variance for the Maxwell-Boltzmann distribution. Defaulting to 1.')
        self.boltzmann_deviation = np.sqrt(boltzmann_variance)

        # dV/dx = 2a(x+(a/b)^1/2). This sets the (a/b)^1/2 term
        try:
            self.root_a_over_b = np.sqrt(constants.a/constants.b)
        except:
            self.root_a_over_b = 1
            print('Cannot set up a proper (a/b)^1/2 term. Defaulting to 1.')

        # sets the 'a' term from above.
        try:
            self.a = constants.a
        except:
            self.a = 1
            print('Cannot set up a proper a value from the constants. Defaulting to 1.')

        # Friction term
        try:
            self.friction = constants.friction
        except:
            self.friction = 0.05
            print('Cannot set up a proper value from the constants. Defaulting to 0.05')

        # position, velocity, force, mass
        self.x = 0
        self.velocity = sample_maxwell_boltzmann_distribution(self.boltzmann_deviation)
        self.initial_velocity = self.velocity
        self.force = self.force_at_x()
        self.m = constants.m

    def force_at_x(self):
        '''
Calculates the force at the current position.

dV/dx = 2a(x+sqrt(a/b))

Returns:
    float: The calculated force at the current position.
        '''
        return (2*self.a)*(self.x+self.root_a_over_b)

    def _sample_thermal_fluctuation(self):
        '''
Samples a thermal fluctuation from a normal distribution.

Returns:
    float: A sampled thermal fluctuation, or 0 if no fluctuation is defined.
        '''
        if self.boltzmann_deviation == 0:
            return 0.0
        else:
            return np.random.normal(0,self.boltzmann_deviation)

    def calculate_langevin_forces(self):
        '''
Calculates the total force on the particle, including external forces, friction, 
and thermal fluctuations based on the Langevin equation.

ma(t+dt) = F(t+dt) = [F(x)-z*v(t)+R(t)]
        '''
        thermal_fluctuation = self._sample_thermal_fluctuation()
        force_external = self.force_at_x()
        self.force = (force_external-self.friction*self.velocity+thermal_fluctuation)

    def propagate_velocity(self,timestep:float=constants.dt):
        '''
Updates the particle's velocity according to the Langevin equation.

v(t+dt) = v(t) + 1/m*F(t)dt

Args:
    timestep (float): The time step for the simulation.
        '''
        self.velocity = (self.velocity + 1/self.m * self.force * timestep)

    def propagage_position(self, timestep:float=constants.dt):
        '''
Updates the particle's position based on the velocity.

Args:
    timestep (float): The time step for the simulation.
        '''
        self.x = self.x + self.velocity*timestep

    def simulate_langevin_motion(self, timestep:float=constants.dt, numsteps:int=constants.steps):
        '''
Simulates Langevin motion over a given number of timesteps, updating the position, 
velocity, and forces.

Args:
    timestep (float): The time step for the simulation.
    numsteps (int): The number of simulation steps to run.

Returns:
    tuple: The final position, trajectory, and initial velocity of the particle.
        '''
        self._set_initial_variables()
        curstep = 0
        while curstep < (numsteps+1):
            self.trajectory.append(self.x)
            self.calculate_langevin_forces()
            self.propagate_velocity(timestep=timestep)
            self.propagage_position(timestep=timestep)
            curstep += 1
        return self.x, self.trajectory, self.initial_velocity

if __name__ == '__main__':
    import matplotlib.pyplot as plt # for ~graphing~

    # instantiate key variables
    sim = LangevinSimulation()
    final_pos = [] # sim final positions
    final_loc = [] # sim final well locations
    initial_velocity = [] # sim initial velocities
    num_sims = 1000 # how many particles do you want to simulate? 1000 sims: ~30s clock time on my laptop at 4 au time, dt=0.001 au.
    num_progress_updates = 10 # how often to tell the user where the sim is right now

    # core logic for running lots of sims
    for i in range(num_sims):
        # send progress updates for the user
        if i % (num_sims // num_progress_updates) == 0:
            print(f'Running Simulation {i+1}.')

        # do an actual simulation
        final_pos_i, traj_i, initial_velocity_i = sim.simulate_langevin_motion(constants.dt, constants.steps)
        final_pos.append(final_pos_i) # store the final position
        final_loc.append(int(final_pos_i>0)) # in a weird way, store whether it's in the product or reactant well
        initial_velocity.append(initial_velocity_i) # save the initial velocity
        time_index = np.linspace(0,constants.time,len(traj_i)) # create an x-coordinate (time)
        plt.plot(time_index, traj_i,linestyle='-',marker='') # add the trajectories to the plot as they come in

    # Just a bunch of plotting

    # axis titles
    plt.xlabel('Time (au)')
    plt.ylabel('Position (au)')

    # horizontal line to give the different wells
    plt.axhline(y=0,color='red',linestyle='--')

    # annotating the well names
    plt.text(0.3, 0.7, 'Product Well', color='black', ha='center', va='bottom', transform=plt.gca().transAxes)
    plt.text(0.3, 0.2, 'Reactant Well', color='black', ha='center', va='top', transform=plt.gca().transAxes)

    # plot title
    plt.title(f'Trajectories ($\zeta$ = {constants.friction}, T = {constants.T}, a = {constants.a}, b = {constants.b})')

    # automatically set scaling on axes
    scaling_val = max(max(final_pos),abs(min(final_pos)))
    plt.ylim(-scaling_val, scaling_val)

    # make sure everything's visible & display to user
    plt.tight_layout()
    plt.show()

    # another plot. this time, show whether things ended up in reactant or product well as a function of their initial velocity
    plt.scatter(final_loc,initial_velocity, s=1) # scatter plot, very small dots

    # axis titles
    plt.xlabel('Final Location (au)')
    plt.ylabel('Initial Velocity (au)')

    # vertical line marking the different wells
    plt.axvline(x=0.5,color='red',linestyle='--')

    # annotate well names
    plt.text(0.7, 0.3, 'Product Well', color='black', ha='center', va='bottom', transform=plt.gca().transAxes)
    plt.text(0.2, 0.3, 'Reactant Well', color='black', ha='center', va='top', transform=plt.gca().transAxes)

    # plot title
    plt.title(f'Ending well by $V_i$ ($\zeta$ = {constants.friction}, T = {constants.T}, a = {constants.a}, b = {constants.b})')

    # again, set scaling value. here, it's in a range [velocity, 0). Also add 0.1 au margin to axes
    scaling_val = max(max(initial_velocity),abs(min(initial_velocity)))
    plt.ylim(-scaling_val-0.1, 0.1)

    # make sure everything's visible and display to user
    plt.tight_layout()
    plt.show()