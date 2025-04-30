import numpy as np
import matplotlib.pyplot as plt
from boltzmann_sampler import sample_maxwell_boltzmann_distribution
import constants
from langevin_simulator import LangevinSimulation  # Import the existing simulation class

def calculate_reactive_flux(trajectories, initial_velocities):
    '''
    Calculate the reactive flux correlation function:
    k_cl = (1/xR)⟨xdot(0)delta[x* - x(0)]hp[x(t)]⟩
    '''
    num_sims = len(trajectories)
    timesteps = len(trajectories[0])
    reactive_flux = np.zeros(timesteps)
    
    for t in range(timesteps):
        weighted_sum = 0
        for i in range(num_sims):
            # Heaviside function: 1 if in product well (x > 0), 0 otherwise
            h_p = 1 if trajectories[i][t] > 0 else 0
            weighted_sum += initial_velocities[i] * h_p
        
        # Normalize by number of simulations and xR (approximated as 0.5 for symmetric wells)
        # Negative sign because we're looking at flux into the product well
        # and initial velocities are negative
        reactive_flux[t] = -weighted_sum / (num_sims * 0.5)
    
    return reactive_flux

def run_simulations_and_calculate_flux():
    '''
    Run Langevin simulations and calculate the reactive flux
    '''
    print("Running simulations for part (d)...")
    
    # Simulation parameters
    num_sims = 1000
    num_progress_updates = 10
    timesteps = int(constants.time / constants.dt) + 1
    
    # Initialize arrays to store results
    all_trajectories = []
    initial_velocities = []
    
    # Run simulations
    for i in range(num_sims):
        if i % (num_sims // num_progress_updates) == 0:
            print(f'Running Simulation {i+1} of {num_sims}')
        
        # Create new simulation instance
        sim = LangevinSimulation()
        
        # Run the simulation
        final_pos, trajectory, initial_vel = sim.simulate_langevin_motion()
        
        # Store results
        all_trajectories.append(trajectory)
        initial_velocities.append(initial_vel)
    
    # Calculate reactive flux
    time_points = np.linspace(0, constants.time, timesteps)
    reactive_flux = calculate_reactive_flux(all_trajectories, initial_velocities)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, reactive_flux)
    
    # Find the plateau value (using average of last 20% of points)
    plateau_start = int(0.8 * timesteps)
    plateau_value = np.mean(reactive_flux[plateau_start:])
    plt.axhline(y=plateau_value, color='r', linestyle='--')
    plt.text(constants.time * 0.7, plateau_value * 1.1, f'Plateau: {plateau_value:.6f}')
    
    plt.xlabel('Time (au)')
    plt.ylabel('Reactive Flux Correlation Function')
    plt.title('Reactive Flux vs Time (Classical Rate Constant)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reactive_flux.png')
    plt.show()
    
    # Plot a few sample trajectories to verify
    plt.figure(figsize=(10, 6))
    for i in range(min(20, num_sims)):  # Plot first 20 trajectories
        plt.plot(time_points, all_trajectories[i], alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.text(constants.time * 0.7, 1.0, 'Product Well (x > 0)')
    plt.text(constants.time * 0.7, -1.0, 'Reactant Well (x < 0)')
    plt.xlabel('Time (au)')
    plt.ylabel('Position (x)')
    plt.title('Sample Particle Trajectories')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sample_trajectories.png')
    plt.show()
    
    print(f"Classical rate constant (k_cl): {plateau_value:.6f}")
    return plateau_value

if __name__ == '__main__':
    # Run the simulations and calculate reactive flux
    k_cl = run_simulations_and_calculate_flux()