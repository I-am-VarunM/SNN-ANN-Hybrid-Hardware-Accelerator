import math
import numpy as np
import matplotlib.pyplot as plt
from sparten_simulator import SparTenSimulator

class SparTenClusterDynamic:
    """
    Simulator for a SparTen cluster with dynamic assignment of work
    to compute units as they finish their current tasks
    """
    
    def __init__(self, num_compute_units=16):
        # Cluster configuration
        self.NUM_COMPUTE_UNITS = num_compute_units
        self.CHUNK_SIZE = 128  # Elements per chunk
        
        # Create the compute units
        self.compute_units = [SparTenSimulator() for _ in range(self.NUM_COMPUTE_UNITS)]
        
        # Scheduler power in mW
        self.SCHEDULER_POWER = 0.132  # mW
        
        # Operating frequency in MHz
        self.FREQ_MHZ = 560
        
        # Performance metrics
        self.total_cycles = 0
        self.total_energy_nj = 0
        self.total_matches = 0
        self.total_chunks = 0
        self.total_rows = 0
        self.partial_sums = []
        self.cu_utilization = []  # Track utilization of each compute unit
        self.pre_relu_sums = []
        # Breakdown of energy consumption
        self.energy_breakdown = {
            "scheduler": 0,
            "compute_units": 0,
            "compute_unit_breakdown": {}
        }
        
        # Detailed execution trace for visualization and debugging
        self.execution_trace = []
    
    def run_simulation(self, filter_data, feature_data):
        """
        Run the SparTen cluster simulation with the given data, properly handling
        scheduling cycles between operations.
        
        Parameters:
        filter_data: Dictionary with "sparse_maps" and "values" (list of sparse maps and their values)
        feature_data: Dictionary with "sparse_maps" and "values" (list of sparse maps and their values)
        """
        # Extract data
        filter_sparse_maps = filter_data["sparse_maps"]
        filter_values = filter_data["values"]
        feature_sparse_maps = feature_data["sparse_maps"]
        feature_values = feature_data["values"]
        
        # Validate input
        if len(filter_sparse_maps) != len(filter_values):
            raise ValueError("Number of filter sparse maps must match number of filter value lists")
        if len(feature_sparse_maps) != len(feature_values):
            raise ValueError("Number of feature sparse maps must match number of feature value lists")
        
        # Calculate total number of rows to process
        total_rows = len(filter_sparse_maps)
        self.total_rows = total_rows
        
        # 1. Vector Analysis and Preparation
        row_info = []
        for i in range(total_rows):
            # Calculate chunks needed
            vector_length = len(filter_sparse_maps[i])
            chunks_needed = math.ceil(vector_length / self.CHUNK_SIZE)
            
            # Count matches in each chunk
            total_matches = 0
            for j in range(chunks_needed):
                start_idx = j * self.CHUNK_SIZE
                end_idx = min(start_idx + self.CHUNK_SIZE, vector_length)
                
                # Count matches in this chunk
                chunk_matches = sum(1 for k in range(start_idx, end_idx) 
                                if filter_sparse_maps[i][k] and feature_sparse_maps[i][k])
                total_matches += chunk_matches
            
            row_info.append({
                "row_idx": i,
                "chunks_needed": chunks_needed,
                "total_matches": total_matches,
                "vector_length": vector_length
            })
        
        # Sort rows by total matches (highest first)
        row_info.sort(key=lambda x: x["total_matches"], reverse=True)
        
        # Initialize simulation state
        current_cycle = 0
        self.total_cycles = 0
        self.total_energy_nj = 0
        self.total_matches = 0
        self.total_chunks = 0
        self.partial_sums = [0] * total_rows
        self.pre_relu_sums = [0] * total_rows
        self.cu_utilization = [0] * self.NUM_COMPUTE_UNITS
        self.execution_trace = []
        
        # Initialize compute unit state tracking
        cu_state = []
        for i in range(self.NUM_COMPUTE_UNITS):
            cu_state.append({
                "busy_until": 0,          # When this CU will finish current task
                "current_row": None,      # Which row is being processed
                "current_chunk": 0,       # Which chunk is being processed
                "total_chunks": 0,        # Total chunks for current row
                "processing_phase": None, # "scheduling", "computing", "relu"
                "accumulated_sum": 0,     # Running sum for current row
                "total_active_cycles": 0  # For utilization calculation
            })
        
        # Row assignment queue
        row_queue = [info["row_idx"] for info in row_info]
        rows_completed = 0
        scheduler_energy = 0
        
        # Initialize event queue for simulation
        # Events are (cycle, cu_idx, event_type) tuples
        event_queue = []
        
        # Initial assignment of rows to compute units
        for cu_idx in range(min(self.NUM_COMPUTE_UNITS, len(row_queue))):
            if row_queue:
                row_idx = row_queue.pop(0)
                
                # Calculate row info
                chunks_needed = math.ceil(len(filter_sparse_maps[row_idx]) / self.CHUNK_SIZE)
                self.total_chunks += chunks_needed
                
                # Schedule first chunk (takes 1 cycle)
                cu_state[cu_idx]["current_row"] = row_idx
                cu_state[cu_idx]["current_chunk"] = 0
                cu_state[cu_idx]["total_chunks"] = chunks_needed
                cu_state[cu_idx]["processing_phase"] = "scheduling"
                cu_state[cu_idx]["busy_until"] = current_cycle + 1  # Scheduling takes 1 cycle
                
                # Add event for scheduling completion
                event_queue.append((current_cycle + 1, cu_idx, "chunk_scheduled"))
                
                # Track scheduler energy
                scheduler_energy += self.SCHEDULER_POWER / self.FREQ_MHZ
                
                # Log event
                self.execution_trace.append({
                    "cycle": current_cycle,
                    "event": f"Scheduling row {row_idx} chunk 0 to CU {cu_idx}",
                    "details": {
                        "total_chunks": chunks_needed
                    }
                })
        
        # Main simulation loop
        while rows_completed < total_rows:
            # No events in queue would mean we're stuck
            if not event_queue:
                break
            
            # Sort events by cycle
            event_queue.sort(key=lambda x: x[0])
            
            # Get next event
            next_cycle, cu_idx, event_type = event_queue.pop(0)
            
            # Advance simulation time
            current_cycle = next_cycle
            
            # Process the event
            if event_type == "chunk_scheduled":
                # Chunk has been scheduled, start computation
                row_idx = cu_state[cu_idx]["current_row"]
                chunk_idx = cu_state[cu_idx]["current_chunk"]
                
                # Calculate start and end indices for this chunk
                vector_length = len(filter_sparse_maps[row_idx])
                start_idx = chunk_idx * self.CHUNK_SIZE
                end_idx = min(start_idx + self.CHUNK_SIZE, vector_length)
                
                # Create chunk-specific data
                chunk_filter_map = filter_sparse_maps[row_idx][start_idx:end_idx]
                chunk_feature_map = feature_sparse_maps[row_idx][start_idx:end_idx]
                
                # Count matches in this chunk
                matches = sum(1 for k in range(len(chunk_filter_map)) 
                            if chunk_filter_map[k] and chunk_feature_map[k])
                
                # Calculate computation time: 4 + (matches - 1)
                compute_cycles = 4 + max(0, matches - 1) if matches > 0 else 1
                
                # Update state
                cu_state[cu_idx]["processing_phase"] = "computing"
                cu_state[cu_idx]["busy_until"] = current_cycle + compute_cycles
                
                # Track active cycles
                cu_state[cu_idx]["total_active_cycles"] += compute_cycles
                
                # Schedule completion event
                event_queue.append((current_cycle + compute_cycles, cu_idx, "chunk_computed"))
                
                # Compute energy and update metrics
                chip_energy = 11.010 * matches * 1/560  # We would calculate this based on component activity
                self.total_energy_nj += chip_energy
                self.total_matches += matches
                
                # Log event
                self.execution_trace.append({
                    "cycle": current_cycle,
                    "event": f"CU {cu_idx} computing row {row_idx} chunk {chunk_idx}",
                    "details": {
                        "matches": matches,
                        "compute_cycles": compute_cycles,
                        "completion_cycle": current_cycle + compute_cycles
                    }
                })
                
            elif event_type == "chunk_computed":
                # Chunk computation complete
                row_idx = cu_state[cu_idx]["current_row"]
                chunk_idx = cu_state[cu_idx]["current_chunk"]
                
                # Calculate chunk contribution to sum
                # (In a real simulation, this would come from the simulator)
                vector_length = len(filter_sparse_maps[row_idx])
                start_idx = chunk_idx * self.CHUNK_SIZE
                end_idx = min(start_idx + self.CHUNK_SIZE, vector_length)
                
                # Extract values based on filter and feature maps
                chunk_filter_map = filter_sparse_maps[row_idx][start_idx:end_idx]
                chunk_feature_map = feature_sparse_maps[row_idx][start_idx:end_idx]
                
                # Find matches
                matches = []
                for i in range(len(chunk_filter_map)):
                    if chunk_filter_map[i] and chunk_feature_map[i]:
                        # Find position in original sparse maps
                        filter_pos = sum(1 for j in range(start_idx + i) if filter_sparse_maps[row_idx][j])
                        feature_pos = sum(1 for j in range(start_idx + i) if feature_sparse_maps[row_idx][j])
                        
                        if filter_pos <= len(filter_values[row_idx]) and feature_pos <= len(feature_values[row_idx]):
                            filter_val = filter_values[row_idx][filter_pos-1]
                            feature_val = feature_values[row_idx][feature_pos-1]
                            matches.append(filter_val * feature_val)
                
                # Update accumulated sum
                chunk_sum = sum(matches)
                cu_state[cu_idx]["accumulated_sum"] += chunk_sum
                
                # Check if more chunks remain
                if chunk_idx + 1 < cu_state[cu_idx]["total_chunks"]:
                    # Schedule next chunk (takes 1 cycle)
                    cu_state[cu_idx]["current_chunk"] += 1
                    cu_state[cu_idx]["processing_phase"] = "scheduling"
                    cu_state[cu_idx]["busy_until"] = current_cycle + 1
                    
                    # Add event for scheduling completion
                    event_queue.append((current_cycle + 1, cu_idx, "chunk_scheduled"))
                    
                    # Track scheduler energy
                    scheduler_energy += self.SCHEDULER_POWER / self.FREQ_MHZ
                    
                    # Log event
                    self.execution_trace.append({
                        "cycle": current_cycle,
                        "event": f"Scheduling row {row_idx} chunk {chunk_idx+1} to CU {cu_idx}",
                        "details": {
                            "accumulated_sum": cu_state[cu_idx]["accumulated_sum"]
                        }
                    })
                else:
                    # All chunks processed, schedule ReLU (takes 1 cycle)
                    cu_state[cu_idx]["processing_phase"] = "relu"
                    cu_state[cu_idx]["busy_until"] = current_cycle + 1
                    
                    # Add event for ReLU completion
                    event_queue.append((current_cycle + 1, cu_idx, "relu_completed"))
                    
                    # Log event
                    self.execution_trace.append({
                        "cycle": current_cycle,
                        "event": f"CU {cu_idx} performing ReLU for row {row_idx}",
                        "details": {
                            "pre_relu_sum": cu_state[cu_idx]["accumulated_sum"]
                        }
                    })
            
            elif event_type == "relu_completed":
                # ReLU computation complete
                row_idx = cu_state[cu_idx]["current_row"]
                
                # Apply ReLU function
                pre_relu_sum = cu_state[cu_idx]["accumulated_sum"]
                post_relu_sum = max(0, pre_relu_sum)
                
                # Store results
                self.pre_relu_sums[row_idx] = pre_relu_sum
                self.partial_sums[row_idx] = post_relu_sum
                
                # Mark row as completed
                rows_completed += 1
                
                # Log event
                self.execution_trace.append({
                    "cycle": current_cycle,
                    "event": f"CU {cu_idx} completed row {row_idx}",
                    "details": {
                        "pre_relu_sum": pre_relu_sum,
                        "post_relu_sum": post_relu_sum
                    }
                })
                
                # Check if more rows are available
                if row_queue:
                    # Schedule next row (takes 1 cycle)
                    new_row_idx = row_queue.pop(0)
                    
                    # Reset compute unit state for new row
                    chunks_needed = math.ceil(len(filter_sparse_maps[new_row_idx]) / self.CHUNK_SIZE)
                    self.total_chunks += chunks_needed
                    
                    cu_state[cu_idx]["current_row"] = new_row_idx
                    cu_state[cu_idx]["current_chunk"] = 0
                    cu_state[cu_idx]["total_chunks"] = chunks_needed
                    cu_state[cu_idx]["processing_phase"] = "scheduling"
                    cu_state[cu_idx]["busy_until"] = current_cycle + 1
                    cu_state[cu_idx]["accumulated_sum"] = 0
                    
                    # Add event for scheduling completion
                    event_queue.append((current_cycle + 1, cu_idx, "chunk_scheduled"))
                    
                    # Track scheduler energy
                    scheduler_energy += self.SCHEDULER_POWER / self.FREQ_MHZ
                    
                    # Log event
                    self.execution_trace.append({
                        "cycle": current_cycle,
                        "event": f"Scheduling row {new_row_idx} chunk 0 to CU {cu_idx}",
                        "details": {
                            "total_chunks": chunks_needed
                        }
                    })
                else:
                    # No more rows, mark compute unit as idle
                    cu_state[cu_idx]["current_row"] = None
                    cu_state[cu_idx]["processing_phase"] = None
        
        # Update final metrics
        self.total_cycles = current_cycle
        self.energy_breakdown["scheduler"] = scheduler_energy
        self.energy_breakdown["compute_units"] = self.total_energy_nj
        self.total_energy_nj += scheduler_energy
        
        # Calculate compute unit utilization
        for cu_idx in range(self.NUM_COMPUTE_UNITS):
            self.cu_utilization[cu_idx] = cu_state[cu_idx]["total_active_cycles"] / max(1, self.total_cycles)
        
        # Return a report of the simulation results
        return self.generate_report()
    
    def generate_report(self):
        """Generate a report of the simulation results"""
        report = {
            "total_cycles": self.total_cycles,
            "total_energy_nj": self.total_energy_nj,
            "energy_per_operation_nj": self.total_energy_nj / max(1, self.total_matches),
            "total_matches": self.total_matches,
            "total_chunks": self.total_chunks,
            "total_rows": self.total_rows,
            "rows_per_cycle": self.total_rows / max(1, self.total_cycles),
            "operations_per_cycle": self.total_matches / max(1, self.total_cycles),
            "partial_sums": self.partial_sums,
            "pre_relu" : self.pre_relu_sums,
            "cu_utilization": self.cu_utilization,
            "average_utilization": sum(self.cu_utilization) / max(1, len(self.cu_utilization)),
            "energy_breakdown": self.energy_breakdown,
            "execution_trace": self.execution_trace
        }
        return report
    
    def print_summary(self, report=None):
        """Print a summary of the simulation results"""
        if report is None:
            report = self.generate_report()
        
        print("\n=== SparTen Cluster Dynamic Simulation Summary ===")
        print(f"Total rows processed: {report['total_rows']}")
        print(f"Total chunks processed: {report['total_chunks']}")
        print(f"Total matches found: {report['total_matches']}")
        print(f"Total cycles: {report['total_cycles']}")
        print(f"Rows per cycle: {report['rows_per_cycle']:.2f}")
        print(f"Operations per cycle: {report['operations_per_cycle']:.2f}")
        print(f"Average compute unit utilization: {report['average_utilization']*100:.1f}%")
        
        # Print individual compute unit utilization
        print("\nCompute unit utilization:")
        for i, util in enumerate(report['cu_utilization']):
            print(f"  CU {i}: {util*100:.1f}%")
        
        # Print individual partial sums
        print("\nPartial sums for each row:")
        for i, partial_sum in enumerate(report['partial_sums']):
            print(f"  Row {i}: {partial_sum}")
        
        # Print summary statistics for partial sums
        # sums = report['partial_sums']
        # if sums:
        #     print(f"\nPartial sums summary statistics:")
        #     print(f"  Average: {sum(sums)/len(sums):.2f}")
        #     print(f"  Minimum: {min(sums)}")
        #     print(f"  Maximum: {max(sums)}")
        #     print(f"  Total: {sum(sums)}")
        
        print(f"\nTotal energy consumption: {report['total_energy_nj']:.2f} nJ")
        print(f"Energy per operation: {report['energy_per_operation_nj']:.2f} nJ")
        
        # Scheduler vs. compute units energy
        scheduler_energy = report['energy_breakdown']['scheduler']
        compute_energy = report['energy_breakdown']['compute_units']
        
        print("\nEnergy breakdown:")
        print(f"  Scheduler: {scheduler_energy:.2f} nJ ({scheduler_energy/report['total_energy_nj']*100:.1f}%)")
        print(f"  Compute Units: {compute_energy:.2f} nJ ({compute_energy/report['total_energy_nj']*100:.1f}%)")
        print("================================================\n")
    
    def visualize_execution(self, report=None):
        """Visualize the execution timeline with proper scheduling cycles and phases"""
        if report is None:
            report = self.generate_report()
        
        trace = report['execution_trace']
        if not trace:
            print("No execution trace available for visualization.")
            return
        
        # Extract event information and track state changes
        cu_activity = {}
        for event in trace:
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
        
        # Create Gantt chart visualization
        plt.figure(figsize=(16, 10))
        
        # Create a colormap for rows
        row_colors = plt.cm.viridis(np.linspace(0, 1, report['total_rows']))
        
        # Phase-specific colors
        phase_colors = {
            'scheduling': 'lightgrey',
            'computing': 'cornflowerblue',
            'relu': 'firebrick'
        }
        
        # Plot compute unit activity
        y_ticks = []
        y_labels = []
        
        for cu_idx, activities in sorted(cu_activity.items()):
            y_pos = cu_idx
            y_ticks.append(y_pos)
            y_labels.append(f"CU {cu_idx}")
            
            for activity in activities:
                duration = activity['end'] - activity['start']
                
                # Set color based on phase
                color = activity['color']
                
                # Add activity bar
                plt.barh(y_pos, duration, left=activity['start'], color=color, 
                        alpha=0.8, edgecolor='black')
                
                # Add label if bar is wide enough
                if duration > report['total_cycles'] * 0.03:
                    if activity['phase'] == 'computing':
                        label = f"R{activity['row']}C{activity.get('chunk', '')}"
                    elif activity['phase'] == 'relu':
                        label = f"R{activity['row']}:ReLU"
                    else:
                        label = f"Sched"
                    
                    plt.text(activity['start'] + duration/2, y_pos, label, 
                            ha='center', va='center', fontsize=8, 
                            color='black', fontweight='bold')
        
        # Add legend for phases
        legend_elements = [
            plt.Rectangle((0,0), 1, 1, facecolor='lightgrey', edgecolor='black', alpha=0.8, label='Scheduling'),
            plt.Rectangle((0,0), 1, 1, facecolor='cornflowerblue', edgecolor='black', alpha=0.8, label='Computing'),
            plt.Rectangle((0,0), 1, 1, facecolor='firebrick', edgecolor='black', alpha=0.8, label='ReLU')
        ]
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Set chart properties
        plt.yticks(y_ticks, y_labels)
        plt.xlabel('Cycle')
        plt.ylabel('Compute Unit')
        plt.title('SparTen Cluster Execution Timeline with Scheduling Phases')
        plt.grid(axis='x', alpha=0.3)
        
        # Add cycle markers at regular intervals
        max_cycle = report['total_cycles']
        cycle_interval = max(1, max_cycle // 20)
        plt.xticks(range(0, max_cycle + cycle_interval, cycle_interval))
        
        plt.tight_layout()
        plt.show()
        
        # Create utilization visualization
        plt.figure(figsize=(12, 6))
        
        cu_indices = range(self.NUM_COMPUTE_UNITS)
        utilization = report['cu_utilization']
        
        plt.bar(cu_indices, utilization)
        plt.axhline(report['average_utilization'], color='r', linestyle='--', 
                    label=f"Average: {report['average_utilization']*100:.1f}%")
        
        plt.xlabel('Compute Unit')
        plt.ylabel('Utilization')
        plt.title('Compute Unit Utilization')
        plt.xticks(cu_indices, [f"CU {i}" for i in cu_indices])
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# Create a simple debugging test case where different rows have very different match counts
def create_debug_test_data():
    """
    Create a test case with rows of varying complexity:
    - Some rows with many matches (heavy rows)
    - Some rows with few matches (light rows)
    """
    rows = 20  # Total number of rows
    elements = 256  # Elements per row
    
    filter_sparse_maps = []
    filter_values = []
    feature_sparse_maps = []
    feature_values = []
    
    # Create rows with varying density
    for i in range(rows):
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

def run_dynamic_cluster_test():
    """Run a test of the SparTen dynamic cluster simulator"""
    print("=== SparTen Dynamic Cluster Test ===")
    
    # Create a cluster
    cluster = SparTenClusterDynamic(num_compute_units=16)
    
    # Create debug test data
    test_data = create_debug_test_data()
    
    filter_data = test_data["filter_data"]
    feature_data = test_data["feature_data"]
    
    # Run simulation
    report = cluster.run_simulation(filter_data, feature_data)
    
    # Print summary
    cluster.print_summary(report)
    
    # Visualize execution
    cluster.visualize_execution(report)
    
    return cluster, report

if __name__ == "__main__":
    cluster, report = run_dynamic_cluster_test()
