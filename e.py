import numpy as np
import matplotlib.pyplot as plt
import constants
from langevin_simulator import LangevinSimulation

def calculate_tst_rate():
    '''
    Calculate the transition state theory rate from part (b)
    k_TST = (1/2π)√(2a/m) × e^(-βa²/4b)
    '''
    prefactor = np.sqrt(2 * constants.a / constants.m) / (2 * np.pi)
    beta = 1 / (constants.k_b * constants.T)
    exponent = -beta * constants.a**2 / (4 * constants.b)
    k_tst = prefactor * np.exp(exponent)
    return k_tst

def run_simulation_with_friction(friction_value, num_sims=1000, plot_results=False, eps=0.05):
    '''
    Run Langevin simulations with a specific friction value and calculate classical rate
    '''
    # Store trajectories and velocities
    all_trajectories = []
    initial_velocities = []
    final_positions = []
    
    # Run simulations
    print(f"  Running {num_sims} simulations with friction ξ = {friction_value:.4f}")
    for _ in range(num_sims):
        # Create new simulation instance
        sim = LangevinSimulation()
        
        # Set custom friction
        sim.friction = friction_value

        
        # Run simulation
        final_pos, trajectory, initial_vel = sim.simulate_langevin_motion()
        
        # Store results
        all_trajectories.append(trajectory)
        initial_velocities.append(initial_vel)
        final_positions.append(final_pos)

    print("Accumulating Flux.")
    velocity_weighted_sum = sum([initial_velocities[i]*int(final_positions[i]>0) for i in range(len(initial_velocities))])

    k_cl = 2*abs(velocity_weighted_sum)/num_sims
    
    # # Calculate reactive flux correlation function
    # timesteps = len(all_trajectories[0])
    # reactive_flux = np.zeros(timesteps)
    
    # for t in range(timesteps):
    #     weighted_sum = 0
    #     for i in range(num_sims):
    #         # Heaviside function: 1 if in product well (x > 0), 0 otherwise
    #         h_product = 1 if all_trajectories[i][t] > 0 else 0
    #         # Weight by initial velocity (negative values)
    #         weighted_sum += initial_velocities[i] * h_product
        
    #     # Normalize by number of simulations and xR (approximated as 0.5 for symmetric wells)
    #     # The negative sign is because initial velocities are negative and we want positive rate
    #     reactive_flux[t] = -weighted_sum / (num_sims * 0.5)
    
    # # Find plateau value (average of last 20% of data)
    # i = int(0.8 * timesteps)
    # while reactive_flux[i]+eps*max(reactive_flux) < max(reactive_flux) and i < timesteps:
    #     i += 1
    # plateau_start = i
    # if i > 0.95*timesteps:
    #     print(f'Bad reactive flux calculation at friction {friction_value}. Extend simulation time')
    # k_cl = np.mean(reactive_flux[plateau_start:])
    
    # # Optional: Plot the reactive flux to verify plateau behavior
    # if plot_results:  # Plot for selected friction values
    #     plt.figure(figsize=(8, 5))
    #     plt.plot(np.linspace(0, constants.time, timesteps), reactive_flux)
    #     plt.axhline(y=k_cl, color='r', linestyle='--')
    #     plt.xlabel('Time (au)')
    #     plt.ylabel('Reactive Flux')
    #     plt.title(f'Reactive Flux for Friction ξ = {friction_value:.2f}')
    #     plt.grid(True, alpha=0.3)
    #     plt.savefig(f'reactive_flux_friction_{friction_value:.1f}.png')
    #     plt.close()
    
    return k_cl

def run_part_e():
    '''
    Run simulations for different friction values and calculate the transmission coefficient κ
    '''
    print("Running simulations for part (e)...")
    
    # Calculate TST rate
    k_tst = calculate_tst_rate()
    print(f"Transition State Theory rate: k_TST = {k_tst:.6e}")
    
    # Array of friction values to test (logarithmic scale)
    num_friction = 100
    friction_values = np.linspace(0.001,10,num_friction)  # From 0.01 to ~31.6
    kappa_values = []
    k_cl_values = []
    
    # Run simulations for each friction value
    for idx, friction in enumerate(friction_values):
        plot = (idx in [0, num_friction//2, num_friction])
        k_cl = run_simulation_with_friction(friction,plot_results=plot)
        k_cl_values.append(k_cl)
        
        # Calculate transmission coefficient
        kappa = k_cl / k_tst
        kappa_values.append(kappa)
        
        print(f"  ξ = {friction:.4f}: k_cl = {k_cl:.6e}, κ = {kappa:.6f}")

    smoothed_kappa = [sum(kappa_values[i:i+4])/4 for i in range(len(kappa_values)-4)]
    corresponding_friction = [sum(friction_values[i:i+4])/4 for i in range(len(friction_values)-4)]
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(friction_values, kappa_values, 'o-', markersize=2, label='Scaling Coefficient')
    plt.plot(corresponding_friction, smoothed_kappa,'-.', label='Two Period Moving Average')
    plt.xlabel('Friction (ξ)')
    plt.ylabel('Transmission Coefficient (κ)')
    plt.title('Transmission Coefficient vs. Friction')
    plt.grid(True, which="both", alpha=0.3)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='$k_{cl}=k_{TST}$')  # Reference line at κ = 1
    plt.legend()
    plt.tight_layout()
    plt.savefig('transmission_coefficient.png')
    #plt.show()
    
    # Save the data
    result_data = np.column_stack((friction_values, k_cl_values, kappa_values))
    np.savetxt('transmission_coefficient_data.txt', 
               result_data, 
               header='Friction\tClassical_Rate\tTransmission_Coefficient', 
               delimiter='\t')
    
    return friction_values, kappa_values

if __name__ == "__main__":
    run_part_e()