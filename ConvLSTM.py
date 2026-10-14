"""
convlstm_model.py — Convolutional LSTM for SST forecasting.

A self-contained ConvLSTM implementation for sequence-to-sequence
prediction on regular lat/lon grids. All architecture hyper-
parameters are read from ``config.cfg.model_training``.

Architecture overview
─────────────────────
  Input (B, T, C_in, H, W)
    │
    ├─ Channel lifting (if needed) → (B, T, embed_dim, H, W)
    │
    ├─ N × ConvLSTM layers
    │     ├─ Input gate (convolution + sigmoid)
    │     ├─ Forget gate (convolution + sigmoid)
    │     ├─ Cell gate (convolution + tanh)
    │     ├─ Output gate (convolution + sigmoid)
    │     └─ Hidden state & Cell state update
    │
    ├─ Channel projection → (B, C_out, H, W)
    │
    └─ (optional) global residual skip from input
"""

import torch
import torch.nn as nn
from config import cfg
from typing import Tuple, Optional, List


# ── ConvLSTM Cell ───────────────────────────────────────────────────

class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell that processes one time step.
    
    Implements the standard LSTM gates using 2D convolutions instead
    of fully connected layers, preserving spatial structure.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int, bias: bool):
        """
        Parameters
        ----------
        input_dim : int
            Number of input channels
        hidden_dim : int
            Number of hidden state channels
        kernel_size : int
            Size of convolutional kernel
        bias : bool
            Whether to use bias in convolutions
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        # Combined convolution for all gates (more efficient)
        # Outputs: [input_gate, forget_gate, cell_gate, output_gate]
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
    
    def forward(self, x: torch.Tensor, 
                hidden_state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, C_in, H, W)
            Input tensor for current time step
        hidden_state : tuple of (h, c)
            h : (B, hidden_dim, H, W) - hidden state
            c : (B, hidden_dim, H, W) - cell state
            
        Returns
        -------
        h_next, c_next : tuple of tensors
            Updated hidden and cell states
        """
        h_cur, c_cur = hidden_state
        
        # Concatenate input and hidden state
        combined = torch.cat([x, h_cur], dim=1)  # (B, input_dim + hidden_dim, H, W)
        
        # Apply convolution and split into gates
        gates = self.conv(combined)
        i, f, g, o = torch.split(gates, self.hidden_dim, dim=1)
        
        # Apply gate activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)     # Cell gate
        o = torch.sigmoid(o)  # Output gate
        
        # Update cell and hidden states
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size: int, image_size: Tuple[int, int], 
                    device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and cell states with zeros."""
        height, width = image_size
        h = torch.zeros(batch_size, self.hidden_dim, height, width, 
                       device=device, dtype=torch.float32)
        c = torch.zeros(batch_size, self.hidden_dim, height, width,
                       device=device, dtype=torch.float32)
        return h, c


# ── Multi-layer ConvLSTM ────────────────────────────────────────────

class ConvLSTM(nn.Module):
    """
    Stacked ConvLSTM layers for sequence processing.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 kernel_sizes: List[int], num_layers: int,
                 bias: bool, return_all_layers: bool):
        """
        Parameters
        ----------
        input_dim : int
            Number of channels in input
        hidden_dims : list of int
            Number of hidden channels for each layer
        kernel_sizes : list of int
            Kernel sizes for each layer
        num_layers : int
            Number of ConvLSTM layers
        bias : bool
            Whether to use bias in convolutions
        return_all_layers : bool
            If True, returns outputs from all layers; else only last layer
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.num_layers = num_layers
        self.bias = bias
        self.return_all_layers = return_all_layers
        
        # Build cells for each layer
        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=hidden_dims[i],
                    kernel_size=kernel_sizes[i],
                    bias=bias
                )
            )
        self.cell_list = nn.ModuleList(cell_list)
    
    def forward(self, x: torch.Tensor, 
                hidden_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None):
        """
        Parameters
        ----------
        x : (B, T, C, H, W)
            Input sequence
        hidden_state : list of (h, c) tuples, optional
            Initial hidden states for each layer
            
        Returns
        -------
        layer_output_list : list of (B, T, hidden_dim, H, W)
            Outputs from each layer (if return_all_layers=True)
        last_state_list : list of (h, c) tuples
            Final hidden states for each layer
        """
        B, T, C, H, W = x.size()
        
        # Initialize hidden states if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=B, image_size=(H, W),
                                            device=x.device)
        
        layer_output_list = []
        last_state_list = []
        
        # Process each layer
        cur_layer_input = x
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            # Process each time step
            for t in range(T):
                h, c = self.cell_list[layer_idx](
                    x=cur_layer_input[:, t, :, :, :],
                    hidden_state=(h, c)
                )
                output_inner.append(h)
            
            # Stack outputs over time dimension
            layer_output = torch.stack(output_inner, dim=1)  # (B, T, hidden_dim, H, W)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append((h, c))
        
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
        
        return layer_output_list, last_state_list
    
    def _init_hidden(self, batch_size: int, image_size: Tuple[int, int],
                     device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Initialize hidden states for all layers."""
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.cell_list[i].init_hidden(batch_size, image_size, device)
            )
        return init_states


# ── Full ConvLSTM Model for SST Forecasting ────────────────────────

class ConvLSTMModel(nn.Module):
    """
    ConvLSTM-based model for global SST forecasting.
    
    Takes a sequence of ``input_history`` SST fields (+ optional features)
    and predicts ``output_channels`` SST fields one time-step ahead.
    
    All parameters are read from ``cfg.model_training``.
    """
    
    def __init__(self):
        super().__init__()
        conf = cfg.model_training
        
        # ── Architecture parameters from config ──────────────────
        c_in = conf.total_input_channels
        c_out = conf.output_channels
        
        # ConvLSTM expects same hidden dims for all layers in basic setup
        # You can modify config to have per-layer hidden dims if needed
        hidden_dims = [conf.embed_dim] * conf.num_layers
        
        # Kernel sizes for each layer (can make configurable)
        # Using odd kernel sizes to maintain spatial dimensions
        kernel_sizes = [3] * conf.num_layers
        
        # ── Input encoder (lift channels if needed) ──────────────
        # If c_in != embed_dim, project to embed_dim
        if c_in != conf.embed_dim:
            self.input_encoder = nn.Sequential(
                nn.Conv2d(c_in, conf.embed_dim, kernel_size=3, padding=1, bias=True),
                nn.GroupNorm(1, conf.embed_dim) if conf.norm_type == "layer_norm" else nn.Identity(),
                self._get_activation(conf.activation),
            )
        else:
            self.input_encoder = nn.Identity()
        
        # ── ConvLSTM layers ──────────────────────────────────────
        self.convlstm = ConvLSTM(
            input_dim=conf.embed_dim,
            hidden_dims=hidden_dims,
            kernel_sizes=kernel_sizes,
            num_layers=conf.num_layers,
            bias=True,
            return_all_layers=False  # Only use last layer output
        )
        
        # ── Output decoder ───────────────────────────────────────
        # Project from hidden_dim back to output channels
        self.decoder = nn.Sequential(
            nn.Conv2d(conf.embed_dim, conf.embed_dim // 2, kernel_size=1, bias=True),
            self._get_activation(conf.activation),
            nn.Dropout2d(conf.dropout_rate),
            nn.Conv2d(conf.embed_dim // 2, c_out, kernel_size=1, bias=True),
        )
        
        # ── Optional global residual ─────────────────────────────
        self.use_residual = conf.use_residual
        if self.use_residual:
            self.residual_proj = nn.Conv2d(1, c_out, kernel_size=1, bias=True)
        
        self._init_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function from config string."""
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(activation, nn.GELU())
    
    def _init_weights(self):
        """Xavier initialization for convolutional layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, 
                hidden_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, C_in, H, W) or (B, C_in, H, W)
            Input tensor:
            - If 4D: single time step, will be unsqueezed to (B, 1, C_in, H, W)
            - If 5D: sequence of T time steps
            C_in includes SST history + optional grid features
        hidden_state : list of (h, c) tuples, optional
            Initial hidden states for ConvLSTM layers
            
        Returns
        -------
        pred : (B, C_out, H, W)
            Predicted SST field(s) for next time step
        """
        # Handle both 4D and 5D inputs
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B, C, H, W) → (B, 1, C, H, W)
        
        B, T, C, H, W = x.shape
        
        # Store most recent frame for residual connection
        # Assuming first channel is most recent SST
        most_recent = x[:, -1, 0:1, :, :]  # (B, 1, H, W)
        
        # ── Encode input ─────────────────────────────────────────
        # Flatten time and batch for encoding
        x_flat = x.view(B * T, C, H, W)
        x_encoded = self.input_encoder(x_flat)
        x_encoded = x_encoded.view(B, T, -1, H, W)
        
        # ── Process through ConvLSTM ─────────────────────────────
        layer_output_list, last_state_list = self.convlstm(x_encoded, hidden_state)
        
        # Get last layer's last time step output
        lstm_out = layer_output_list[-1][:, -1, :, :, :]  # (B, hidden_dim, H, W)
        
        # ── Decode to output ─────────────────────────────────────
        out = self.decoder(lstm_out)  # (B, C_out, H, W)
        
        # ── Apply residual connection ────────────────────────────
        if self.use_residual:
            out = out + self.residual_proj(most_recent)
        
        return out
    
    def forecast(self, x: torch.Tensor, num_steps: int) -> torch.Tensor:
        """
        Autoregressive forecasting for multiple steps ahead.
        
        Parameters
        ----------
        x : (B, T, C_in, H, W)
            Initial input sequence
        num_steps : int
            Number of steps to forecast
            
        Returns
        -------
        predictions : (B, num_steps, C_out, H, W)
            Forecasted SST fields
        """
        predictions = []
        hidden_state = None
        
        # Use initial sequence to warm up the model
        current_input = x
        
        for step in range(num_steps):
            # Predict next step
            pred = self.forward(current_input, hidden_state)
            predictions.append(pred)
            
            # Prepare next input
            # Shift sequence: drop oldest, add newest prediction
            # If using grid features, need to append them to prediction
            conf = cfg.model_training
            
            if conf.add_grid_features or conf.add_seasonal_encoding:
                # Extract non-SST features from last time step
                B, T, C, H, W = current_input.shape
                extra_features = current_input[:, -1, 1:, :, :]  # (B, C-1, H, W)
                
                # Concatenate prediction with features
                next_frame = torch.cat([pred, extra_features], dim=1)  # (B, C, H, W)
            else:
                next_frame = pred
            
            # Update input sequence
            next_frame = next_frame.unsqueeze(1)  # (B, 1, C, H, W)
            current_input = torch.cat([current_input[:, 1:, :, :, :], next_frame], dim=1)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=1)  # (B, num_steps, C_out, H, W)
        return predictions
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Grid features helper (same as SFNO) ────────────────────────────

def make_grid_features(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    """
    Build normalised 2-D lat/lon grids as extra input channels.
    
    Parameters
    ----------
    lat : 1-D tensor of latitude values
    lon : 1-D tensor of longitude values
    
    Returns
    -------
    grid : (1, 2, len(lat), len(lon))
    """
    lat_norm = lat / 90.0          # → [-1, 1]
    lon_norm = lon / 180.0 - 1.0   # → [-1, 1]
    lat_2d, lon_2d = torch.meshgrid(lat_norm, lon_norm, indexing="ij")
    grid = torch.stack([lat_2d, lon_2d], dim=0).unsqueeze(0)  # (1, 2, H, W)
    return grid