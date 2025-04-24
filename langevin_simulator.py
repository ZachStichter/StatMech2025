'''
Implements the Langevin equation for a particle with inital position x and velocity x'
'''
from boltzmann_sampler import sample_maxwell_boltzmann_distribution
import numpy as np
import constants

# Set up std deviation for thermal distribution
try:
    second_moment = constants.friction/(constants.T*constants.k_b)
except:
    second_moment = 0
    print('Cannot set a correct second moment of the correlation distribution. Defaulting to 0.')
thermal_fluctuation_deviation = np.sqrt(second_moment)

# Set up std deviation for maxwell-boltzmann distribution
try:
    boltzmann_variance = np.sqrt(constants.k_b*constants.T/constants.m)
except:
    boltzmann_variance = 1
    print('Cannot set up a proper variance for the Maxwell-Boltzmann distribution. Defaulting to 1.')
boltzmann_deviation = np.sqrt(boltzmann_variance)

def set_initial_variables():
    global position
    global velocity
    global force
    global boltzmann_variance
    position = 0
    velocity = sample_maxwell_boltzmann_distribution(boltzmann_variance)
    force = 0

def sample_thermal_fluctuation(std_deviation:float):
    try:
        assert std_deviation >= 0
    except AssertionError:
        raise ValueError('Second moment must be greater than zero by definition.')
    if std_deviation == 0:
        return 0.0
    else:
        return np.random.normal(0,std_deviation)

def calculate_langevin_forces(force:float, friction:float=constants.friction, timestep:float=constants.dt):
    '''
    F(t) = -F(t-dt)-z*v(t-dt)dt+R(t); z is friction, F is force, v & R are normal
    '''
    global thermal_fluctuation_deviation
    thermal_fluctuation = sample_thermal_fluctuation(thermal_fluctuation_deviation)
    return (-force-friction*velocity*timestep+thermal_fluctuation)

def propagate_velocity(force:float, mass:float=constants.m, timestep:float=constants.dt):
    '''
    v(t) = v(t-1) + 1/m*F(t)dt
    '''
    global velocity
    return (velocity + 1/mass * force * timestep)

def propagage_position(timestep:float=constants.dt):
    global position
    global velocity
    return (position + velocity*timestep)

def simulate_langevin_motion(timestep:float=constants.dt, numsteps:int=constants.steps):
    global force
    global velocity
    global position
    trajectory = []
    curstep = 0
    while curstep < (numsteps+1):
        #print(f'Step {curstep}: {position}')
        force = calculate_langevin_forces(force,timestep=timestep)
        trajectory.append(int(force>0))
        propagate_velocity(force,timestep=timestep)
        position = propagage_position(timestep=timestep)
        curstep += 1
    return position, trajectory

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    results = []
    traj_hist = []
    set_initial_variables()
    for i in range(100):
        set_initial_variables()
        end_pos, traj = simulate_langevin_motion()
        results.append(end_pos)
        average_location = sum(traj)/constants.steps
        traj_hist.append(average_location)
    print(min(results))
    plt.hist(results,bins=100)
    plt.show()
    plt.hist(traj_hist,bins=100)
    plt.show()