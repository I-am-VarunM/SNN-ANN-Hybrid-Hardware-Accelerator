import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from collections import OrderedDict

class ANNtoSNNConverter(nn.Module):
    """
    Converts ANN activations to SNN spike trains
    Corresponds to the ann_to_snn_converter Verilog module
    With optional 8-bit quantization
    """
    def __init__(self, time_steps=4, threshold=1.0, quantize=False, scale_factor=1.0):
        """
        Args:
            time_steps: Number of time steps to simulate
            threshold: Membrane potential threshold for spiking
            quantize: Whether to quantize to int8
            scale_factor: Scale factor for quantization
        """
        super(ANNtoSNNConverter, self).__init__()
        self.time_steps = time_steps
        self.threshold = threshold
        self.quantize = quantize
        self.scale_factor = scale_factor
        self.reset()
        
    def reset(self):
        """Reset internal state"""
        # Initialize membrane potential to half the threshold (as in Verilog)
        self.membrane = None
        self.spikes = None
        self.t_counter = 0
        
    def quantize_to_int8(self, x):
        """
        Quantize values to int8 range (-128 to 127)
        """
        if not self.quantize:
            return x
            
        # Scale the input 
        x_scaled = x / self.scale_factor
        
        # Clamp to int8 range
        x_clamped = torch.clamp(x_scaled, -128, 127)
        
        # Quantize by rounding to nearest integer
        x_quantized = torch.round(x_clamped)
        
        # Scale back
        return x_quantized * self.scale_factor
        
    def forward(self, x):
        """
        Convert ANN activation to SNN spike train
        
        Args:
            x: Input tensor with shape [batch_size, channels, height, width]
               representing ANN activation values
               
        Returns:
            spike_out: Tensor with shape [batch_size, time_steps, channels, height, width]
                      containing spike trains
        """
        batch_size, channels, height, width = x.shape
        
        # Quantize input if needed
        if self.quantize:
            x = self.quantize_to_int8(x)
        
        # Initialize membrane potential if not exists
        if self.membrane is None:
            self.membrane = torch.ones_like(x) * (self.threshold / 2)
            self.spikes = torch.zeros(batch_size, self.time_steps, channels, height, width, 
                                     device=x.device, dtype=torch.bool)
        
        # Create temporal dimension by repeating input
        # This simulates the x_buffer in Verilog
        x_temporal = x.unsqueeze(1).repeat(1, self.time_steps, 1, 1, 1)
        
        # Process each timestep (similar to PROCESS state in Verilog)
        for t in range(self.time_steps):
            # Update membrane potential
            self.membrane = self.membrane + x_temporal[:, t]
            
            # Quantize membrane potential if needed
            if self.quantize:
                self.membrane = self.quantize_to_int8(self.membrane)
            
            # Check if membrane exceeds threshold
            spike_occurred = self.membrane >= self.threshold
            
            # Store spikes
            self.spikes[:, t] = spike_occurred
            
            # Reset membrane potential where spikes occurred
            self.membrane[spike_occurred] -= self.threshold
        
        # Convert boolean spikes to float for compatibility
        return self.spikes.float()

class SNNtoANNConverter(nn.Module):
    """
    Converts SNN spike trains to ANN activations
    Corresponds to the snn_to_ann_single_neuron Verilog module
    """
    def __init__(self, time_steps=4):
        """
        Args:
            time_steps: Number of time steps in the spike train
        """
        super(SNNtoANNConverter, self).__init__()
        self.time_steps = time_steps
        
    def forward(self, spikes):
        """
        Convert SNN spike train to ANN activation by summing spikes across time
        
        Args:
            spikes: Tensor with shape [batch_size, time_steps, channels, height, width]
                   containing spike trains
                   
        Returns:
            ann_out: Tensor with shape [batch_size, channels, height, width]
                    representing ANN activation values
        """
        # Sum across the time dimension (axis=1)
        # This is equivalent to the tree adder structure in Verilog
        return torch.sum(spikes, dim=1)

class SpikingNeuron(nn.Module):
    """
    Spiking neuron with leaky integrate-and-fire (LIF) dynamics
    with optional 8-bit quantization
    """
    def __init__(self, threshold=1.0, leak_factor=0.0, quantize=False, scale_factor=1.0):
        """
        Args:
            threshold: Membrane potential threshold for spiking
            leak_factor: Membrane potential leak factor (0.0 = no leak)
            quantize: Whether to quantize membrane potential to int8
            scale_factor: Scale factor for quantization (membrane / scale_factor) before quantizing
        """
        super(SpikingNeuron, self).__init__()
        self.threshold = threshold
        self.leak_factor = leak_factor
        self.quantize = quantize
        self.scale_factor = scale_factor
        self.reset_state()
        
    def reset_state(self):
        """Reset membrane potential and spike history"""
        self.membrane = None
        
    def quantize_to_int8(self, x):
        """
        Quantize values to int8 range (-128 to 127)
        """
        if not self.quantize:
            return x
            
        # Scale the input (typically membrane potentials are in range [0, threshold])
        x_scaled = x / self.scale_factor
        
        # Clamp to int8 range
        x_clamped = torch.clamp(x_scaled, -128, 127)
        
        # Quantize by rounding to nearest integer
        x_quantized = torch.round(x_clamped)
        
        # Scale back
        return x_quantized * self.scale_factor
        
    def forward(self, x, output_spikes=True):
        """
        Forward pass for spiking neuron
        
        Args:
            x: Input tensor
            output_spikes: If True, output binary spikes; otherwise, membrane potential
            
        Returns:
            Spike tensor or membrane potential tensor
        """
        batch_size = x.shape[0]
        
        # Quantize input if needed
        if self.quantize:
            x = self.quantize_to_int8(x)
        
        # Initialize membrane potential if not exists
        if self.membrane is None:
            self.membrane = torch.zeros_like(x)
        
        # Apply leak
        if self.leak_factor > 0:
            self.membrane = self.membrane * (1 - self.leak_factor)
            
        # Update membrane potential
        self.membrane = self.membrane + x
        
        # Quantize membrane potential if needed
        if self.quantize:
            self.membrane = self.quantize_to_int8(self.membrane)
        
        # Generate spikes if membrane potential exceeds threshold
        spike = (self.membrane >= self.threshold).float()
        
        # Reset membrane potential where spikes occurred
        self.membrane = self.membrane - spike * self.threshold
        
        if output_spikes:
            return spike
        else:
            return self.membrane

class HybridLayer(nn.Module):
    """
    A hybrid layer that can be run as either ANN or SNN
    with optional 8-bit quantization
    """
    def __init__(self, 
                 layer: nn.Module,
                 mode: str = 'ann',
                 time_steps: int = 4,
                 threshold: float = 1.0,
                 leak_factor: float = 0.0,
                 quantize: bool = False,
                 scale_factor: float = 1.0):
        """
        Args:
            layer: The PyTorch layer (Conv2d, Linear, etc.)
            mode: 'ann' or 'snn'
            time_steps: Number of time steps to simulate in SNN mode
            threshold: Membrane potential threshold for spiking
            leak_factor: Membrane potential leak factor
            quantize: Whether to quantize to int8
            scale_factor: Scale factor for quantization
        """
        super(HybridLayer, self).__init__()
        self.layer = layer
        self.mode = mode
        self.time_steps = time_steps
        self.threshold = threshold
        self.leak_factor = leak_factor
        self.quantize = quantize
        self.scale_factor = scale_factor
        
        # Extract the base layer without activation for SNN mode
        self.base_layer = self._extract_base_layer(layer)
        
        # SNN components
        self.neuron = SpikingNeuron(threshold=threshold, leak_factor=leak_factor)
        
        # Converters
        self.ann_to_snn = ANNtoSNNConverter(time_steps=time_steps, threshold=threshold)
        self.snn_to_ann = SNNtoANNConverter(time_steps=time_steps)
        
    def _extract_base_layer(self, layer):
        """
        Extract the base layer without activation functions
        for use in SNN mode where activations are replaced by spiking neurons
        """
        if isinstance(layer, nn.Sequential):
            new_layers = []
            for sublayer in layer:
                if not isinstance(sublayer, nn.ReLU) and not isinstance(sublayer, nn.Dropout):
                    new_layers.append(sublayer)
            return nn.Sequential(*new_layers)
        return layer
        
    def reset_state(self):
        """Reset states of SNN components"""
        self.neuron.reset_state()
        self.ann_to_snn.reset()
        
    def forward(self, x, prev_mode=None, convert_output=False, target_mode=None):
        """
        Forward pass through the hybrid layer
        
        Args:
            x: Input tensor
            prev_mode: Mode of the previous layer ('ann' or 'snn')
            convert_output: Whether to convert the output to the other format
            target_mode: Target mode for the output conversion
            
        Returns:
            Output tensor and its mode ('ann' or 'snn')
        """
        batch_size = x.shape[0]
        
        # Input conversion if needed
        if prev_mode is not None and prev_mode != self.mode:
            if self.mode == 'ann' and prev_mode == 'snn':
                # Convert SNN input to ANN
                x = self.snn_to_ann(x)
            elif self.mode == 'snn' and prev_mode == 'ann':
                # Convert ANN input to SNN
                x = self.ann_to_snn(x)
        
        # Process based on mode
        if self.mode == 'ann':
            # Standard ANN forward pass
            out = self.layer(x)
            out_mode = 'ann'
        
        else:  # SNN mode
            # For SNN, handle temporal dimension
            if len(x.shape) == 4:  # [batch, channels, height, width]
                # No temporal dimension yet, create it
                x = x.unsqueeze(1).repeat(1, self.time_steps, 1, 1, 1)
            
            # Process each timestep
            time_steps = x.shape[1]
            spikes_out = []
            
            for t in range(time_steps):
                # Get input for current timestep
                x_t = x[:, t]
                
                # Forward through the base layer WITHOUT activation functions
                # In SNN mode, we use the base_layer (without ReLU)
                x_t = self.base_layer(x_t)
                
                # Forward through spiking neuron (this replaces ReLU in SNN mode)
                spike = self.neuron(x_t)
                spikes_out.append(spike)
            
            # Stack outputs along time dimension
            out = torch.stack(spikes_out, dim=1)
            out_mode = 'snn'
            
        # Output conversion if needed
        if convert_output or (target_mode is not None and target_mode != out_mode):
            if out_mode == 'ann' and (target_mode == 'snn' or target_mode is None):
                # Convert ANN output to SNN
                out = self.ann_to_snn(out)
                out_mode = 'snn'
            elif out_mode == 'snn' and (target_mode == 'ann' or target_mode is None):
                # Convert SNN output to ANN
                out = self.snn_to_ann(out)
                out_mode = 'ann'
                
        return out, out_mode
    
    def set_mode(self, mode):
        """Set layer mode to 'ann' or 'snn'"""
        assert mode in ['ann', 'snn'], "Mode must be 'ann' or 'snn'"
        self.mode = mode
        return self

class HybridVGG16(nn.Module):
    """
    VGG16 network with hybrid SNN-ANN capabilities
    with optional 8-bit quantization
    """
    def __init__(self, num_classes=1000, time_steps=4, threshold=1.0, init_mode='ann', 
                 quantize=False, scale_factor=1.0):
        """
        Args:
            num_classes: Number of output classes
            time_steps: Number of time steps for SNN components
            threshold: Membrane potential threshold for spiking
            init_mode: Initial mode for all layers ('ann' or 'snn')
            quantize: Whether to quantize activations to int8
            scale_factor: Scale factor for quantization
        """
        super(HybridVGG16, self).__init__()
        self.time_steps = time_steps
        self.threshold = threshold
        self.init_mode = init_mode
        self.quantize = quantize
        self.scale_factor = scale_factor
        
        # First layer always runs as ANN (as specified)
        self.first_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Build VGG16 architecture with hybrid layers
        self.hybrid_layers = nn.ModuleList([
            # Block 1 (first conv already defined above)
            HybridLayer(
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                ),
                mode=init_mode,
                time_steps=time_steps,
                threshold=threshold,
                quantize=quantize,
                scale_factor=scale_factor
            ),
            
            # Block 2
            HybridLayer(
                nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True)
                ),
                mode=init_mode,
                time_steps=time_steps,
                threshold=threshold
            ),
            HybridLayer(
                nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True)
                ),
                mode=init_mode,
                time_steps=time_steps,
                threshold=threshold
            ),
            
            # Block 3
            HybridLayer(
                nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                ),
                mode=init_mode,
                time_steps=time_steps,
                threshold=threshold
            ),
            HybridLayer(
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                ),
                mode=init_mode,
                time_steps=time_steps,
                threshold=threshold
            ),
            HybridLayer(
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                ),
                mode=init_mode,
                time_steps=time_steps,
                threshold=threshold
            ),
            
            # Block 4
            HybridLayer(
                nn.Sequential(
                    nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True)
                ),
                mode=init_mode,
                time_steps=time_steps,
                threshold=threshold
            ),
            HybridLayer(
                nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True)
                ),
                mode=init_mode,
                time_steps=time_steps,
                threshold=threshold
            ),
            HybridLayer(
                nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True)
                ),
                mode=init_mode,
                time_steps=time_steps,
                threshold=threshold
            ),
            
            # Block 5
            HybridLayer(
                nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True)
                ),
                mode=init_mode,
                time_steps=time_steps,
                threshold=threshold
            ),
            HybridLayer(
                nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True)
                ),
                mode=init_mode,
                time_steps=time_steps,
                threshold=threshold
            ),
            HybridLayer(
                nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True)
                ),
                mode=init_mode,
                time_steps=time_steps,
                threshold=threshold
            ),
            
            # Fully connected layers
            HybridLayer(
                nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout()
                ),
                mode=init_mode,
                time_steps=time_steps,
                threshold=threshold
            ),
            HybridLayer(
                nn.Sequential(
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout()
                ),
                mode=init_mode,
                time_steps=time_steps,
                threshold=threshold
            ),
            HybridLayer(
                nn.Sequential(
                    nn.Linear(4096, num_classes)
                ),
                mode=init_mode,
                time_steps=time_steps,
                threshold=threshold
            )
        ])
        
        # Pooling layers (not hybrid, applied after specific hybrid layers)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # After layer 1
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # After layer 3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # After layer 6
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # After layer 9
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # After layer 12
        
        # Converters for handling mode transitions
        self.ann_to_snn = ANNtoSNNConverter(time_steps=time_steps, threshold=threshold,
                                          quantize=quantize, scale_factor=scale_factor)
        self.snn_to_ann = SNNtoANNConverter(time_steps=time_steps)
        
        # Layer mode configuration
        self.layer_modes = [init_mode] * len(self.hybrid_layers)
        
    def reset_states(self):
        """Reset states of all SNN components"""
        for layer in self.hybrid_layers:
            layer.reset_state()
            
    def set_layer_mode(self, layer_idx, mode):
        """
        Set mode for a specific layer
        
        Args:
            layer_idx: Index of the layer to configure
            mode: 'ann' or 'snn'
        """
        assert 0 <= layer_idx < len(self.hybrid_layers), f"Layer index out of range: {layer_idx}"
        assert mode in ['ann', 'snn'], "Mode must be 'ann' or 'snn'"
        
        self.hybrid_layers[layer_idx].set_mode(mode)
        self.layer_modes[layer_idx] = mode
        
    def set_all_modes(self, mode):
        """Set all layers to the same mode"""
        for i in range(len(self.hybrid_layers)):
            self.set_layer_mode(i, mode)
            
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input image tensor [batch_size, channels, height, width]
            
        Returns:
            Model output tensor
        """
        batch_size = x.shape[0]
        
        # First layer is always ANN
        x = self.first_layer(x)
        current_mode = 'ann'
        
        # Process through all hybrid layers with pooling at appropriate positions
        for i, layer in enumerate(self.hybrid_layers):
            # Apply pooling at specific positions
            if i == 1:  # After first block
                if current_mode == 'ann':
                    x = self.pool1(x)
                else:  # SNN mode
                    x = torch.stack([self.pool1(x[:, t]) for t in range(x.shape[1])], dim=1)
            elif i == 3:  # After second block
                if current_mode == 'ann':
                    x = self.pool2(x)
                else:  # SNN mode
                    x = torch.stack([self.pool2(x[:, t]) for t in range(x.shape[1])], dim=1)
            elif i == 6:  # After third block
                if current_mode == 'ann':
                    x = self.pool3(x)
                else:  # SNN mode
                    x = torch.stack([self.pool3(x[:, t]) for t in range(x.shape[1])], dim=1)
            elif i == 9:  # After fourth block
                if current_mode == 'ann':
                    x = self.pool4(x)
                else:  # SNN mode
                    x = torch.stack([self.pool4(x[:, t]) for t in range(x.shape[1])], dim=1)
            elif i == 12:  # After fifth block
                if current_mode == 'ann':
                    x = self.pool5(x)
                    # Flatten for FC layers
                    x = torch.flatten(x, 1)
                else:  # SNN mode
                    x = torch.stack([self.pool5(x[:, t]) for t in range(x.shape[1])], dim=1)
                    # Flatten for FC layers, preserving batch and time dimensions
                    x = x.view(batch_size, x.shape[1], -1)
            
            # Forward through the hybrid layer
            x, current_mode = layer(x, prev_mode=current_mode)
            
        return x

def convert_ann_to_snn(ann_model, target_model=None, time_steps=4, threshold=1.0):
    """
    Convert a pre-trained ANN model to SNN
    
    Args:
        ann_model: Pre-trained ANN model
        target_model: Target SNN model structure (optional)
        time_steps: Number of time steps for SNN
        threshold: Neuron threshold for SNN
        
    Returns:
        Converted SNN model
    """
    if target_model is None:
        # Create a new hybrid VGG16 with all layers in SNN mode
        target_model = HybridVGG16(
            num_classes=1000,
            time_steps=time_steps,
            threshold=threshold,
            init_mode='snn'
        )
    
    # Copy weights from ANN model to corresponding layers in target model
    # This assumes similar structures between the models
    with torch.no_grad():
        # Copy first layer (always ANN)
        target_model.first_layer[0].weight.copy_(ann_model.features[0].weight)
        target_model.first_layer[0].bias.copy_(ann_model.features[0].bias)
        
        # Copy hybrid layers
        feature_layer_idx = 2  # Start after first layer
        hybrid_layer_idx = 0
        
        while feature_layer_idx < len(ann_model.features):
            # Skip non-conv layers (BatchNorm, ReLU, MaxPool)
            if isinstance(ann_model.features[feature_layer_idx], nn.Conv2d):
                # Copy weights to corresponding hybrid layer
                target_model.hybrid_layers[hybrid_layer_idx].layer[0].weight.copy_(
                    ann_model.features[feature_layer_idx].weight
                )
                target_model.hybrid_layers[hybrid_layer_idx].layer[0].bias.copy_(
                    ann_model.features[feature_layer_idx].bias
                )
                hybrid_layer_idx += 1
            
            feature_layer_idx += 1
        
        # Copy classifier layers
        classifier_layer_idx = 0
        
        for i in range(hybrid_layer_idx, len(target_model.hybrid_layers)):
            if classifier_layer_idx < len(ann_model.classifier):
                if isinstance(ann_model.classifier[classifier_layer_idx], nn.Linear):
                    target_model.hybrid_layers[i].layer[0].weight.copy_(
                        ann_model.classifier[classifier_layer_idx].weight
                    )
                    target_model.hybrid_layers[i].layer[0].bias.copy_(
                        ann_model.classifier[classifier_layer_idx].bias
                    )
                classifier_layer_idx += 1
    
    return target_model

# Example usage
def load_pretrained_weights(hybrid_model, pretrained_model):
    """
    Load weights from a pretrained VGG16 model into the hybrid model
    
    Args:
        hybrid_model: HybridVGG16 model to load weights into
        pretrained_model: Pretrained torchvision VGG16 model
        
    Returns:
        hybrid_model with loaded weights
    """
    print("Loading pretrained weights...")
    
    # Transfer first layer weights (always ANN)
    hybrid_model.first_layer[0].weight.data.copy_(pretrained_model.features[0].weight.data)
    hybrid_model.first_layer[0].bias.data.copy_(pretrained_model.features[0].bias.data)
    
    # Handle BatchNorm layers if present in first layer
    if len(hybrid_model.first_layer) > 1 and isinstance(hybrid_model.first_layer[1], nn.BatchNorm2d):
        if isinstance(pretrained_model.features[1], nn.BatchNorm2d):
            hybrid_model.first_layer[1].weight.data.copy_(pretrained_model.features[1].weight.data)
            hybrid_model.first_layer[1].bias.data.copy_(pretrained_model.features[1].bias.data)
            hybrid_model.first_layer[1].running_mean.copy_(pretrained_model.features[1].running_mean)
            hybrid_model.first_layer[1].running_var.copy_(pretrained_model.features[1].running_var)
    
    # Mapping from pretrained feature layers to hybrid layers
    feature_idx = 2  # Start after first layer (conv+bn)
    hybrid_idx = 0
    
    # Transfer weights for convolutional layers
    while feature_idx < len(pretrained_model.features) and hybrid_idx < len(hybrid_model.hybrid_layers):
        if isinstance(pretrained_model.features[feature_idx], nn.Conv2d):
            # For each Conv layer, transfer weights to the corresponding hybrid layer
            # Find the conv layer within the hybrid layer
            conv_layer = None
            if isinstance(hybrid_model.hybrid_layers[hybrid_idx].layer, nn.Sequential):
                for sublayer in hybrid_model.hybrid_layers[hybrid_idx].layer:
                    if isinstance(sublayer, nn.Conv2d):
                        conv_layer = sublayer
                        break
            elif isinstance(hybrid_model.hybrid_layers[hybrid_idx].layer, nn.Conv2d):
                conv_layer = hybrid_model.hybrid_layers[hybrid_idx].layer
            
            # Also check the base_layer (used in SNN mode)
            if conv_layer is None and isinstance(hybrid_model.hybrid_layers[hybrid_idx].base_layer, nn.Sequential):
                for sublayer in hybrid_model.hybrid_layers[hybrid_idx].base_layer:
                    if isinstance(sublayer, nn.Conv2d):
                        conv_layer = sublayer
                        break
            elif conv_layer is None and isinstance(hybrid_model.hybrid_layers[hybrid_idx].base_layer, nn.Conv2d):
                conv_layer = hybrid_model.hybrid_layers[hybrid_idx].base_layer
                        
            if conv_layer is not None:
                # Copy weights and bias
                conv_layer.weight.data.copy_(pretrained_model.features[feature_idx].weight.data)
                conv_layer.bias.data.copy_(pretrained_model.features[feature_idx].bias.data)
                print(f"Transferred weights for convolutional layer {feature_idx} to hybrid layer {hybrid_idx}")
            
            # Handle BatchNorm if present in pretrained model
            if (feature_idx + 1 < len(pretrained_model.features) and 
                isinstance(pretrained_model.features[feature_idx + 1], nn.BatchNorm2d)):
                
                # Find the BatchNorm layer in the hybrid layer
                bn_layer = None
                if isinstance(hybrid_model.hybrid_layers[hybrid_idx].layer, nn.Sequential):
                    for sublayer in hybrid_model.hybrid_layers[hybrid_idx].layer:
                        if isinstance(sublayer, nn.BatchNorm2d):
                            bn_layer = sublayer
                            break
                
                # Also check the base_layer
                if bn_layer is None and isinstance(hybrid_model.hybrid_layers[hybrid_idx].base_layer, nn.Sequential):
                    for sublayer in hybrid_model.hybrid_layers[hybrid_idx].base_layer:
                        if isinstance(sublayer, nn.BatchNorm2d):
                            bn_layer = sublayer
                            break
                
                if bn_layer is not None:
                    bn_layer.weight.data.copy_(pretrained_model.features[feature_idx + 1].weight.data)
                    bn_layer.bias.data.copy_(pretrained_model.features[feature_idx + 1].bias.data)
                    bn_layer.running_mean.copy_(pretrained_model.features[feature_idx + 1].running_mean)
                    bn_layer.running_var.copy_(pretrained_model.features[feature_idx + 1].running_var)
                    print(f"Transferred BatchNorm parameters for layer {feature_idx+1}")
                        
            hybrid_idx += 1
        
        feature_idx += 1
    
    # Transfer weights for classifier layers (fully connected layers)
    classifier_indices = [i for i, layer in enumerate(pretrained_model.classifier) 
                         if isinstance(layer, nn.Linear)]
    
    fc_layers_start = hybrid_idx  # First FC layer index in hybrid_layers
    
    # Map classifier layers to the remaining hybrid layers
    for i, cls_idx in enumerate(classifier_indices):
        if fc_layers_start + i >= len(hybrid_model.hybrid_layers):
            break
            
        # Find the Linear layer in the hybrid layer
        linear_layer = None
        if isinstance(hybrid_model.hybrid_layers[fc_layers_start + i].layer, nn.Sequential):
            for sublayer in hybrid_model.hybrid_layers[fc_layers_start + i].layer:
                if isinstance(sublayer, nn.Linear):
                    linear_layer = sublayer
                    break
        elif isinstance(hybrid_model.hybrid_layers[fc_layers_start + i].layer, nn.Linear):
            linear_layer = hybrid_model.hybrid_layers[fc_layers_start + i].layer
        
        # Also check the base_layer
        if linear_layer is None and isinstance(hybrid_model.hybrid_layers[fc_layers_start + i].base_layer, nn.Sequential):
            for sublayer in hybrid_model.hybrid_layers[fc_layers_start + i].base_layer:
                if isinstance(sublayer, nn.Linear):
                    linear_layer = sublayer
                    break
        elif linear_layer is None and isinstance(hybrid_model.hybrid_layers[fc_layers_start + i].base_layer, nn.Linear):
            linear_layer = hybrid_model.hybrid_layers[fc_layers_start + i].base_layer
                    
        if linear_layer is not None:
            # Copy weights and bias
            linear_layer.weight.data.copy_(pretrained_model.classifier[cls_idx].weight.data)
            linear_layer.bias.data.copy_(pretrained_model.classifier[cls_idx].bias.data)
            print(f"Transferred weights for fully connected layer {cls_idx} to hybrid layer {fc_layers_start + i}")
    
    print("Weight transfer complete!")
    return hybrid_model

def example_with_pretrained_weights():
    """
    Example demonstrating how to load pretrained weights into the hybrid model
    """
    import torchvision.models as models
    
    # Load pretrained VGG16 model
    print("Loading pretrained VGG16 model...")
    pretrained_vgg16 = models.vgg16(pretrained=True)
    
    # Create hybrid model
    print("Creating hybrid model...")
    hybrid_model = HybridVGG16(
        num_classes=1000,
        time_steps=8,
        init_mode='ann',
        quantize=True,
        scale_factor=1.0
    )
    
    # Transfer weights
    hybrid_model = load_pretrained_weights(hybrid_model, pretrained_vgg16)
    
    # Configure some layers as SNN
    hybrid_model.set_layer_mode(1, 'snn')  # Second layer to SNN
    hybrid_model.set_layer_mode(2, 'snn')  # Third layer to SNN
    
    # Reset SNN states
    hybrid_model.reset_states()
    
    print("Model ready for inference!")
    return hybrid_model

if __name__ == "__main__":
    # Example of creating and using a hybrid model with pretrained weights
    model = example_with_pretrained_weights()
    
    # Example inference
    x = torch.randn(1, 3, 224, 224)  # Example input
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Example: Calculate model size
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {param_count:,}")
    
    # Example: Calculate quantized model size (if all parameters were quantized to int8)
    # Each int8 value takes 1 byte
    quantized_size_bytes = param_count
    print(f"Quantized model size: {quantized_size_bytes/1024/1024:.2f} MB")
    
    # Compare with full precision model (each parameter as float32, 4 bytes)
    full_precision_size = param_count * 4
    print(f"Full precision model size: {full_precision_size/1024/1024:.2f} MB")
    print(f"Size reduction: {full_precision_size/quantized_size_bytes:.2f}x")