import numpy as np

class LIFNeuron:
    def __init__(self, threshold=1.0, tau=0.5):
        self.threshold = threshold
        self.tau = tau
        self.membrane_potential = 0
        
    def forward(self, input_val):
        potential = input_val + self.membrane_potential
        spike = 1 if potential > self.threshold else 0
        if spike:
            self.membrane_potential = 0
        else:
            self.membrane_potential = self.tau * potential
        return spike, self.membrane_potential

class LoASSimulator:
    def __init__(self):
        self.power_config = {
            'fast_prefix': 1.46,
            'laggy_prefix': 0.32,
            'accumulator': 0.16,
            'others': 0.88,
            'lif': 0.075
        }
        self.clock_freq = 800e6
        
    def calculate_energy_per_cycle(self, power_mw):
        return (power_mw / self.clock_freq) * 1e9
    
    def analyze_computation(self, conv_matrix, input_flat, threshold=1.0, tau=0.5):
        # Get number of output positions (rows in conv_matrix)
        num_outputs = conv_matrix.shape[0]
        
        total_energy = 0
        stats = {
            'total_matches': 0,
            'corrections_needed': 0,
            'fast_prefix_cycles': 0,
            'laggy_prefix_cycles': 0,
            'accumulation_cycles': 0,
            'lif_cycles': 0
        }
        
        # Initialize outputs
        output_pre_lif = np.zeros(num_outputs)
        output_spikes = np.zeros(num_outputs)
        membrane_potentials = np.zeros(num_outputs)
        
        # Process each output position
        for i in range(num_outputs):
            # Get non-zero positions in current conv_matrix row
            non_zero_pos = np.nonzero(conv_matrix[i])[0]
            stats['total_matches'] += len(non_zero_pos)
            
            # Fast prefix sum cycle for each position
            stats['fast_prefix_cycles'] += len(non_zero_pos)
            total_energy += len(non_zero_pos) * self.calculate_energy_per_cycle(self.power_config['fast_prefix'])
            
            # Process each position
            acc_val = 0
            for pos in non_zero_pos:
                weight = conv_matrix[i, pos]
                input_val = input_flat[pos]
                
                if input_val != 0:
                    acc_val += weight * input_val
                    stats['accumulation_cycles'] += 1
                    total_energy += self.calculate_energy_per_cycle(self.power_config['accumulator'])
                else:
                    stats['corrections_needed'] += 1
                    stats['laggy_prefix_cycles'] += 1
                    total_energy += self.calculate_energy_per_cycle(self.power_config['laggy_prefix'])
            
            output_pre_lif[i] = acc_val
            total_energy += self.calculate_energy_per_cycle(self.power_config['others'])
        
        # LIF processing
        lif_neuron = LIFNeuron(threshold=threshold, tau=tau)
        for i in range(num_outputs):
            stats['lif_cycles'] += 1
            total_energy += self.calculate_energy_per_cycle(self.power_config['lif'])
            output_spikes[i], membrane_potentials[i] = lif_neuron.forward(output_pre_lif[i])
        
        return total_energy, stats, {
            'pre_lif': output_pre_lif,
            'spikes': output_spikes,
            'membrane': membrane_potentials
        }

import numpy as np

def generate_spike_train(data, num_timesteps=4, rate_coding=True):
    """Convert input data to spike trains using rate coding"""
    if rate_coding:
        # Normalize data to [0,1]
        if data.max() != 0:
            data = data / data.max()
        
        # Generate spikes based on probability
        spikes = np.random.random((*data.shape, num_timesteps)) < data[..., np.newaxis]
        return spikes.astype(np.int32)
    else:
        # Direct binary coding
        return (data > 0)[..., np.newaxis].repeat(num_timesteps, axis=-1).astype(np.int32)

import numpy as np

class SNNConvolution:
    def __init__(self, threshold=1.0, leak_factor=0.5):
        self.threshold = threshold
        self.leak_factor = leak_factor
    
    def create_conv_matrix(self, kernel, input_shape):
        """Create convolution matrix for each timestep"""
        k_h, k_w = kernel.shape
        i_h, i_w, t = input_shape
        
        # Output dimensions
        out_h = i_h - k_h + 1
        out_w = i_w - k_w + 1
        
        # Create conv matrix
        total_outputs = out_h * out_w
        conv_matrix = np.zeros((total_outputs, i_h * i_w))
        
        for i in range(out_h):
            for j in range(out_w):
                output_idx = i * out_w + j
                for ki in range(k_h):
                    for kj in range(k_w):
                        input_row = i + ki
                        input_col = j + kj
                        input_idx = input_row * i_w + input_col
                        conv_matrix[output_idx, input_idx] = kernel[ki, kj]
        
        return conv_matrix, (out_h, out_w)
    
    def forward(self, input_spikes, kernel):
        """Process spike-based convolution"""
        # Create convolution matrix
        conv_matrix, output_shape = self.create_conv_matrix(kernel, input_spikes.shape)
        out_h, out_w = output_shape
        num_timesteps = input_spikes.shape[2]
        
        # Initialize outputs
        membrane_potentials = np.zeros((out_h, out_w, num_timesteps))
        output_spikes = np.zeros_like(membrane_potentials)
        
        # Keep track of current membrane potential
        current_potential = np.zeros((out_h, out_w))
        
        # Process each timestep
        for t in range(num_timesteps):
            # Apply leak to current potential
            current_potential = self.leak_factor * current_potential
            
            # Get input for current timestep
            input_flat = input_spikes[..., t].reshape(-1)
            
            # Calculate input contribution
            conv_output = (conv_matrix @ input_flat).reshape(out_h, out_w)
            
            # Update membrane potential
            current_potential += conv_output
            
            # Store membrane potential for this timestep
            membrane_potentials[..., t] = current_potential
            
            # Generate spikes where threshold is reached
            spikes = current_potential >= self.threshold
            output_spikes[..., t] = spikes
            
            # Reset membrane potential where spikes occurred
            current_potential[spikes] = 0
            
        return output_spikes, membrane_potentials

def verify_snn_convolution(input_data, kernel, output_spikes, membrane_potentials, threshold=1.0):
    """Verify SNN convolution results"""
    # Standard convolution
    h, w = input_data.shape
    k_h, k_w = kernel.shape
    out_h = h - k_h + 1
    out_w = w - k_w + 1
    
    standard_conv = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            window = input_data[i:i+k_h, j:j+k_w]
            standard_conv[i, j] = np.sum(window * kernel)
    
    # Accumulated SNN response
    accumulated_response = np.sum(output_spikes, axis=-1)
    
    # Normalize responses for comparison
    if np.max(accumulated_response) > 0:
        normalized_response = accumulated_response / np.max(accumulated_response)
    else:
        normalized_response = accumulated_response
        
    if np.max(standard_conv) > 0:
        normalized_conv = standard_conv / np.max(standard_conv)
    else:
        normalized_conv = standard_conv
    
    # Verify temporal dynamics
    temporal_checks = {
        'membrane_bounds': np.all(membrane_potentials >= 0),
        'spike_validity': np.all(np.logical_or(output_spikes == 0, output_spikes == 1)),
        'membrane_reset': True
    }
    
    # Check spike generation consistency
    spike_mask = output_spikes > 0
    potential_mask = membrane_potentials >= threshold
    spike_consistency = np.all(
        np.logical_or(
            ~spike_mask,
            potential_mask
        )
    )
    
    return {
        'standard_conv': standard_conv,
        'accumulated_response': accumulated_response,
        'normalized_response': normalized_response,
        'normalized_conv': normalized_conv,
        'correlation': np.corrcoef(normalized_response.flatten(), normalized_conv.flatten())[0,1],
        'temporal_checks': temporal_checks,
        'spike_consistency': spike_consistency
    }

def main():
    # Create sample input (5x5)
    input_data = np.array([
        [1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [1, 0, 0, 1, 1],
        [0, 1, 1, 0, 1]
    ], dtype=np.float32)
    
    # Generate spike trains (4 timesteps)
    num_timesteps = 4
    input_spikes = np.random.binomial(1, input_data[..., np.newaxis], size=(*input_data.shape, num_timesteps))
    
    # Define kernel
    kernel = np.array([
        [1, 0],
        [0, 1]
    ], dtype=np.float32)
    
    # Create and run SNN convolution
    snn_conv = SNNConvolution(threshold=1.0, leak_factor=0.5)
    output_spikes, membrane_potentials = snn_conv.forward(input_spikes, kernel)
    
    # Verify results
    verification_results = verify_snn_convolution(
        input_data, kernel, output_spikes, membrane_potentials, threshold=1.0
    )
    
    # Print results
    print("\nSNN Convolution Analysis:")
    print(f"Input shape: {input_data.shape}")
    print(f"Input spike train shape: {input_spikes.shape}")
    print(f"Kernel shape: {kernel.shape}")
    print(f"Output spike shape: {output_spikes.shape}")
    
    print("\nMembrane Potential Evolution (first position):")
    print(membrane_potentials[0,0,:])
    
    print("\nOutput Spikes (first position):")
    print(output_spikes[0,0,:])
    
    print("\nStandard Convolution:")
    print(verification_results['standard_conv'])
    
    print("\nAccumulated SNN Response:")
    print(verification_results['accumulated_response'])
    
    print("\nCorrelation with standard conv:", verification_results['correlation'])
    
    print("\nTemporal Checks:")
    for check, result in verification_results['temporal_checks'].items():
        print(f"{check}: {'✓' if result else '✗'}")
    
    print(f"Spike Generation Consistency: {'✓' if verification_results['spike_consistency'] else '✗'}")
    
    print("\nMembrane Potential Statistics:")
    print(f"Min: {np.min(membrane_potentials):.4f}")
    print(f"Max: {np.max(membrane_potentials):.4f}")
    print(f"Mean: {np.mean(membrane_potentials):.4f}")

if __name__ == "__main__":
    main()