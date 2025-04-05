import numpy as np
from tppe_simulator import TPPESimulator
from TPPE_cluster import TPPEClusterSimulator
from sparten_simulator import SparTenSimulator
from sparten_cluster import SparTenClusterDynamic
import matplotlib.pyplot as plt

class NeuroFlex:
    """
    NeuroFlex: A framework for simulating both LoAS and SparTen neural network accelerators in parallel
    
    This class runs simulations on both hardware accelerator architectures simultaneously
    and reports combined performance metrics.
    """
    
    def __init__(self, num_compute_units=16, clock_freq_mhz=560):
        """
        Initialize the NeuroFlex simulator with both LoAS and SparTen clusters
        
        Parameters:
        - num_compute_units: Number of compute units/TPPEs to simulate
        - clock_freq_mhz: Clock frequency in MHz
        """
        self.num_compute_units = num_compute_units
        self.clock_freq_mhz = clock_freq_mhz
        
        # Initialize both simulators
        self.loas_simulator = TPPEClusterSimulator(num_compute_units, clock_freq_mhz)
        self.sparten_simulator = SparTenClusterDynamic(num_compute_units)
        
        # Results storage
        self.loas_results = None
        self.sparten_results = None
        self.combined_results = None
    
    def run_simulation(self, loas_data, sparten_data):
        """
        Run simulation on both accelerators in parallel
        
        Parameters:
        - loas_data: A tuple of (activations_batch, weights_batch, thresholds) for LoAS
        - sparten_data: A tuple of (filter_data, feature_data) for SparTen
        
        Returns:
        - Combined simulation results
        """
        # Run LoAS simulation
        activations_batch, weights_batch, thresholds = loas_data
        print(f"Running LoAS simulation with {len(activations_batch)} vectors...")
        self.loas_results = self.loas_simulator.simulate(activations_batch, weights_batch, thresholds)
        
        # Run SparTen simulation
        filter_data, feature_data = sparten_data
        print(f"Running SparTen simulation with {len(filter_data['sparse_maps'])} rows...")
        self.sparten_results = self.sparten_simulator.run_simulation(filter_data, feature_data)
        
        # Combine results
        self.combined_results = self._combine_results()
        
        return self.combined_results
    
    def _combine_results(self):
        """Combine results from both simulators"""
        if self.loas_results is None or self.sparten_results is None:
            raise ValueError("Both simulations must be run before combining results")
        
        # Get total cycles (max of both)
        loas_cycles = self.loas_results["Total Cycles"]
        sparten_cycles = self.sparten_results["total_cycles"]
        total_cycles = max(loas_cycles, sparten_cycles)
        
        # Get total energy (sum of both)
        loas_energy = self.loas_results["Energy Metrics"]["Total Cluster Energy (nJ)"]
        sparten_energy = self.sparten_results["total_energy_nj"]
        total_energy = loas_energy + sparten_energy
        
        # Calculate utilization for LoAS
        loas_active_cycles = self.loas_results["TPPE Active Cycles"]
        loas_util = sum(loas_active_cycles) / (self.num_compute_units * loas_cycles) if loas_cycles > 0 else 0
        
        # Get SparTen utilization
        sparten_util = self.sparten_results["average_utilization"]
        
        return {
            "Total Cycles": total_cycles,
            "Total Energy (nJ)": total_energy,
            "LoAS Cycles": loas_cycles,
            "SparTen Cycles": sparten_cycles,
            "LoAS Energy (nJ)": loas_energy,
            "SparTen Energy (nJ)": sparten_energy,
            "LoAS Utilization": loas_util,
            "SparTen Utilization": sparten_util,
            "LoAS Results": self.loas_results,
            "SparTen Results": self.sparten_results
        }
    
    def print_summary(self):
        """Print a summary of the combined simulation results"""
        if self.combined_results is None:
            print("No simulation results available. Run a simulation first.")
            return
        
        print("\n=== NeuroFlex Simulation Summary ===")
        print(f"Total Cycles: {self.combined_results['Total Cycles']}")
        print(f"Total Energy: {self.combined_results['Total Energy (nJ)']:.4f} nJ")
        
        print("\nLoAS Performance:")
        print(f"  Cycles: {self.combined_results['LoAS Cycles']}")
        print(f"  Energy: {self.combined_results['LoAS Energy (nJ)']:.4f} nJ")
        print(f"  Utilization: {self.combined_results['LoAS Utilization']*100:.2f}%")
        
        print("\nSparTen Performance:")
        print(f"  Cycles: {self.combined_results['SparTen Cycles']}")
        print(f"  Energy: {self.combined_results['SparTen Energy (nJ)']:.4f} nJ")
        print(f"  Utilization: {self.combined_results['SparTen Utilization']*100:.2f}%")
        
        # Critical path analysis
        critical_path = "LoAS" if self.combined_results['LoAS Cycles'] >= self.combined_results['SparTen Cycles'] else "SparTen"
        print(f"\nCritical Path: {critical_path} (determining total cycle count)")
        
        print("======================================")
    
    def visualize_execution(self):
        """Visualize the execution timeline of both simulations"""
        if self.loas_results is None or self.sparten_results is None:
            print("No simulation results available. Run a simulation first.")
            return
        
        # Visualize LoAS execution
        print("\n=== LoAS Execution Visualization ===")
        self.loas_simulator.plot_gantt_chart(
            self.loas_results['TPPE Timeline'],
            [{"id": i, "matches": result["Matches"]} for i, result in enumerate(self.loas_results['TPPE Results'])]
        )
        
        # Visualize SparTen execution
        print("\n=== SparTen Execution Visualization ===")
        self.sparten_simulator.visualize_execution(self.sparten_results)
    
    def plot_comparison(self):
        """Create a comparative visualization of LoAS vs SparTen performance"""
        if self.combined_results is None:
            print("No simulation results available. Run a simulation first.")
            return
        
        # Create comparison bar charts
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot cycles comparison
        accelerators = ['LoAS', 'SparTen', 'Combined']
        cycles = [
            self.combined_results['LoAS Cycles'],
            self.combined_results['SparTen Cycles'],
            self.combined_results['Total Cycles']
        ]
        
        axs[0].bar(accelerators, cycles, color=['blue', 'orange', 'green'])
        axs[0].set_title('Execution Cycles Comparison')
        axs[0].set_ylabel('Cycles')
        axs[0].grid(axis='y', alpha=0.3)
        
        # Plot energy comparison
        energy = [
            self.combined_results['LoAS Energy (nJ)'],
            self.combined_results['SparTen Energy (nJ)'],
            self.combined_results['Total Energy (nJ)']
        ]
        
        axs[1].bar(accelerators, energy, color=['blue', 'orange', 'green'])
        axs[1].set_title('Energy Consumption Comparison')
        axs[1].set_ylabel('Energy (nJ)')
        axs[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def visualize_combined_execution(self):
        """
        Visualize both LoAS and SparTen execution timelines on the same chart
        to directly compare their parallel execution.
        """
        if self.combined_results is None:
            print("No simulation results available. Run a simulation first.")
            return
        
        # Create a large figure to accommodate both accelerators
        plt.figure(figsize=(18, 12))
        
        # Get timeline data
        loas_timeline = self.loas_results['TPPE Timeline']
        sparten_timeline = self.sparten_results['execution_trace']
        
        # Determine max cycle for proper x-axis scaling
        max_cycle = max(
            self.combined_results['LoAS Cycles'],
            self.combined_results['SparTen Cycles']
        )
        
        # Parse SparTen timeline to extract activities
        cu_activity = {}
        for event in sparten_timeline:
            cycle = event['cycle']
            event_desc = event['event']
            
            # Parse events to determine CU activities
            if "Scheduling row" in event_desc:
                # Extract row, chunk, and CU info
                parts = event_desc.split()
                row_idx = int(parts[2])
                cu_idx = int(parts[-1])
                
                # Determine if this is first chunk or subsequent chunk
                if "chunk 0" in event_desc:
                    # New row assignment
                    if cu_idx not in cu_activity:
                        cu_activity[cu_idx] = []
                    
                    cu_activity[cu_idx].append({
                        'row': row_idx,
                        'phase': 'scheduling',
                        'start': cycle,
                        'end': cycle + 1,  # Scheduling takes 1 cycle
                        'color': 'lightgrey'
                    })
                else:
                    # Inter-chunk scheduling
                    cu_activity[cu_idx].append({
                        'row': row_idx,
                        'phase': 'scheduling',
                        'start': cycle,
                        'end': cycle + 1,  # Scheduling takes 1 cycle
                        'color': 'lightgrey'
                    })
            
            elif "computing row" in event_desc:
                # Extract row, chunk, and CU info
                parts = event_desc.split()
                cu_idx = int(parts[1])
                row_idx = int(parts[4])
                chunk_idx = int(parts[6])
                
                # Get computation end time
                completion_cycle = event['details']['completion_cycle']
                
                cu_activity[cu_idx].append({
                    'row': row_idx,
                    'phase': 'computing',
                    'chunk': chunk_idx,
                    'start': cycle,
                    'end': completion_cycle,
                    'color': 'cornflowerblue',
                    'matches': event['details']['matches'] if 'matches' in event['details'] else 0
                })
            
            elif "performing ReLU" in event_desc:
                # Extract row and CU info
                parts = event_desc.split()
                cu_idx = int(parts[1])
                row_idx = int(parts[6])
                
                cu_activity[cu_idx].append({
                    'row': row_idx,
                    'phase': 'relu',
                    'start': cycle,
                    'end': cycle + 1,  # ReLU takes 1 cycle
                    'color': 'firebrick'
                })
        
        # Create vector info for LoAS visualization from results
        vector_info = [{"id": i, "matches": result["Matches"]} 
                    for i, result in enumerate(self.loas_results['TPPE Results'])]
        
        # Set up number of rows in the gantt chart
        total_rows = self.num_compute_units * 2  # TPPEs + CUs
        y_ticks = []
        y_labels = []
        
        # Create a subplot
        ax = plt.gca()
        
        # Define colors for different phases
        loas_colors = plt.cm.viridis(np.linspace(0, 1, len(vector_info)))
        sparten_phase_colors = {
            'scheduling': 'lightgrey',
            'computing': 'cornflowerblue',
            'relu': 'firebrick'
        }
        
        # Group LoAS timeline entries by vector and TPPE
        loas_segments = {}
        for tppe_id, vector_id, start, end, is_chunk in loas_timeline:
            key = (tppe_id, vector_id)
            if key not in loas_segments:
                loas_segments[key] = []
            loas_segments[key].append((start, end, is_chunk))
        
        # Plot LoAS activities (first half of the chart)
        for tppe_id in range(self.num_compute_units):
            y_pos = tppe_id
            y_ticks.append(y_pos)
            y_labels.append(f"LoAS TPPE {tppe_id}")
            
            # Plot TPPE activities
            for (tppe_id_seg, vector_id), segments in loas_segments.items():
                if tppe_id_seg == tppe_id:
                    # Sort segments by start time
                    segments.sort(key=lambda x: x[0])
                    
                    for i, (start, end, is_chunk) in enumerate(segments):
                        width = end - start
                        color = loas_colors[vector_id % len(loas_colors)]
                        
                        # Use different patterns for chunks vs LIF
                        if is_chunk:
                            # Chunk segments - solid fill
                            rect = plt.Rectangle((start, y_pos - 0.4), width, 0.8, 
                                            facecolor=color, edgecolor='black', alpha=0.7)
                        else:
                            # LIF segment - hatched pattern
                            rect = plt.Rectangle((start, y_pos - 0.4), width, 0.8, 
                                            facecolor=color, edgecolor='black', 
                                            hatch='////', alpha=0.7)
                        
                        ax.add_patch(rect)
                        
                        # Only add a label on the first segment of each vector
                        if i == 0:
                            label_x = start + 0.5
                            label_y = y_pos
                            plt.text(label_x, label_y, f"V{vector_id}", 
                                    ha='center', va='center', color='white', 
                                    fontweight='bold', fontsize=7)
        
        # Plot SparTen activities (second half of the chart)
        for cu_idx, activities in sorted(cu_activity.items()):
            y_pos = self.num_compute_units + cu_idx  # Offset to place after TPPEs
            y_ticks.append(y_pos)
            y_labels.append(f"SparTen CU {cu_idx}")
            
            for activity in activities:
                duration = activity['end'] - activity['start']
                
                # Set color based on phase
                color = activity['color']
                
                # Add activity bar
                rect = plt.Rectangle((activity['start'], y_pos - 0.4), duration, 0.8, 
                                facecolor=color, edgecolor='black', alpha=0.7)
                ax.add_patch(rect)
                
                # Add label if bar is wide enough
                if duration > max_cycle * 0.02:
                    if activity['phase'] == 'computing':
                        label = f"R{activity['row']}C{activity.get('chunk', '')}"
                    elif activity['phase'] == 'relu':
                        label = f"R{activity['row']}:ReLU"
                    else:
                        label = f"Sched"
                    
                    plt.text(activity['start'] + duration/2, y_pos, label, 
                            ha='center', va='center', fontsize=7, 
                            color='black', fontweight='bold')
        
        # Add legends
        loas_chunk_patch = plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.7)
        loas_lif_patch = plt.Rectangle((0, 0), 1, 1, facecolor='gray', hatch='////', alpha=0.7)
        
        sparten_sched_patch = plt.Rectangle((0, 0), 1, 1, facecolor='lightgrey', alpha=0.7)
        sparten_comp_patch = plt.Rectangle((0, 0), 1, 1, facecolor='cornflowerblue', alpha=0.7)
        sparten_relu_patch = plt.Rectangle((0, 0), 1, 1, facecolor='firebrick', alpha=0.7)
        
        plt.legend([loas_chunk_patch, loas_lif_patch, sparten_sched_patch, sparten_comp_patch, sparten_relu_patch], 
                ['LoAS Computation', 'LoAS LIF', 'SparTen Scheduling', 'SparTen Computing', 'SparTen ReLU'], 
                loc='upper right')
        
        # Add separator line between LoAS and SparTen sections
        plt.axhline(y=self.num_compute_units - 0.5, color='black', linestyle='--', alpha=0.5)
        
        # Add critical path line
        critical_path_cycle = self.combined_results['Total Cycles']
        plt.axvline(x=critical_path_cycle, color='red', linestyle='--', alpha=0.7, 
                label=f'Critical Path ({critical_path_cycle} cycles)')
        
        # Set chart properties
        plt.yticks(y_ticks, y_labels)
        plt.xlabel('Cycles')
        plt.title('NeuroFlex: Combined LoAS and SparTen Execution Timeline')
        plt.grid(axis='x', alpha=0.3)
        
        # Add cycle markers at regular intervals
        cycle_interval = max(1, max_cycle // 20)
        plt.xticks(range(0, max_cycle + cycle_interval, cycle_interval))
        
        # Ensure x-axis extends to the max cycle
        plt.xlim(-1, max_cycle + 1)
        
        # Adjust the y-axis limits to show all units with some padding
        plt.ylim(bottom=-1, top=total_rows)
        
        plt.tight_layout()
        plt.show()


# Test data generation functions
def generate_loas_test_data(num_vectors=40, vector_size=256, timesteps=4):
    """
    Generate test data for LoAS simulator based on the provided pattern
    """
    activations_batch = []
    weights_batch = []
    thresholds = []
    
    test_vectors = []
    for i in range(num_vectors):
        # Create test vector description with varied matches
        if i % 5 == 0:  # Very dense
            matches = 25 + np.random.randint(0, 6)  # 25-30 matches
        elif i % 5 == 1:  # Dense
            matches = 20 + np.random.randint(0, 6)  # 20-25 matches
        elif i % 5 == 2:  # Medium
            matches = 15 + np.random.randint(0, 6)  # 15-20 matches
        elif i % 5 == 3:  # Sparse
            matches = 10 + np.random.randint(0, 6)  # 10-15 matches
        else:  # Very sparse
            matches = 5 + np.random.randint(0, 6)   # 5-10 matches
        
        test_vectors.append({
            "size": vector_size,
            "matches": matches,
            "id": i
        })
    
    for vector in test_vectors:
        size = vector["size"]
        matches = vector["matches"]
        
        # Create activation matrix with exactly 'matches' non-zero positions
        activations = np.zeros((timesteps, size), dtype=int)
        
        # Distribute matches across the activation matrix
        match_positions = np.random.choice(size, matches, replace=False)
        for pos in match_positions:
            # Create a pattern of 1s (can be different for each position)
            pattern = np.random.randint(0, 2, size=timesteps)
            # Ensure at least one 1 in the pattern
            if np.sum(pattern) == 0:
                pattern[np.random.randint(0, timesteps)] = 1
            activations[:, pos] = pattern
        
        # Create weights for this vector
        weights = np.random.randint(1, 11, size=size)
        
        # Set threshold
        threshold = 15
        
        activations_batch.append(activations)
        weights_batch.append(weights)
        thresholds.append(threshold)
    
    return activations_batch, weights_batch, thresholds

def generate_sparten_test_data(num_rows=40, elements=256):
    """
    Generate test data for SparTen simulator based on the provided pattern
    """
    filter_sparse_maps = []
    filter_values = []
    feature_sparse_maps = []
    feature_values = []
    
    # Create rows with varying density
    for i in range(num_rows):
        # Vary density based on row index
        if i % 5 == 0:  # 20% of rows are very dense
            density = 0.8
        elif i % 5 == 1:  # 20% of rows are dense
            density = 0.6
        elif i % 5 == 2:  # 20% of rows are medium
            density = 0.4
        elif i % 5 == 3:  # 20% of rows are sparse
            density = 0.2
        else:  # 20% of rows are very sparse
            density = 0.1
        
        # Create filter with this density
        filter_map = [False] * elements
        for j in range(elements):
            if np.random.random() < density:
                filter_map[j] = True
        
        # Create feature with same density pattern to ensure matches
        feature_map = filter_map.copy()  # Use same pattern for debugging clarity
        
        # Count non-zeros
        filter_nonzeros = sum(filter_map)
        feature_nonzeros = sum(feature_map)
        
        # Create values
        filter_row_values = [i * 10 + j for j in range(filter_nonzeros)]
        feature_row_values = [j + 1 for j in range(feature_nonzeros)]
        
        # Add to lists
        filter_sparse_maps.append(filter_map)
        filter_values.append(filter_row_values)
        feature_sparse_maps.append(feature_map)
        feature_values.append(feature_row_values)
    
    return {
        "filter_data": {
            "sparse_maps": filter_sparse_maps,
            "values": filter_values
        },
        "feature_data": {
            "sparse_maps": feature_sparse_maps,
            "values": feature_values
        }
    }


def run_neuroflex_test():
    """Run a test of the NeuroFlex simulator with both accelerators"""
    print("=== NeuroFlex Test ===")
    
    # Create a NeuroFlex instance
    neuroflex = NeuroFlex(num_compute_units=16)
    
    # Generate test data for LoAS and SparTen
    print("Generating test data...")
    loas_data = generate_loas_test_data(num_vectors=40)
    sparten_data_dict = generate_sparten_test_data(num_rows=40)
    
    # Extract SparTen data
    sparten_data = (
        sparten_data_dict["filter_data"],
        sparten_data_dict["feature_data"]
    )
    
    # Run simulation
    print("Running parallel simulation...")
    results = neuroflex.run_simulation(loas_data, sparten_data)
    
    # Print summary
    neuroflex.print_summary()
    
    # Generate comparison plot
    neuroflex.plot_comparison()
    
    # Visualize execution (optional, can be commented out if not needed)
    neuroflex.visualize_execution()

    neuroflex.visualize_combined_execution()
    
    return neuroflex, results

if __name__ == "__main__":
    neuroflex, results = run_neuroflex_test()
