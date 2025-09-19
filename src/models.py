import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from antialiased_cnns import BlurPool
from .utils import dropout_no_scaling

# Neural network models for attention and representation learning

class Autoencoder(nn.Module):
    def __init__(self,device):
        super().__init__()

        self.device=device
        self.act=nn.LeakyReLU(negative_slope=0.01)


        self.game_res = [84, 84, 1]
        self.n_frame_stacks = 4
        
        # Encoder layers with anti-aliased downsampling
        self.enc_cnn_layer1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            BlurPool(64, stride=2),
            nn.BatchNorm2d(64),
            self.act
        )
        self.enc_cnn_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            BlurPool(64, stride=2),
            nn.BatchNorm2d(64),
            self.act
        )
        self.enc_cnn_layer3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32)
        )
        
        # Decoder layers
        self.dec_cnn_layer1 = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            self.act
        )
        self.dec_cnn_layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            self.act
        )
        self.dec_cnn_layer3 = nn.Sequential(
            nn.ConvTranspose2d(64, 4, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        self.apply(self._weights_init)


    def _weights_init(self, m):
        """Apply Xavier initialization to layers."""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def give_repr_size(self):
        """Get representation size by running a dummy forward pass."""
        a = torch.rand(1, self.game_res[0], self.game_res[1], 
                      self.game_res[2] * self.n_frame_stacks, device=self.device)
        a = self.encode(a, False)
        return len(a.flatten()), a.shape
    
    def encode(self, x, flatten=True):
        """Encode input to representation."""
        x = x.permute(0, 3, 1, 2)
        x = self.enc_cnn_layer1(x)
        x = self.enc_cnn_layer2(x)
        x = self.enc_cnn_layer3(x)
        if flatten:
            x = x.flatten(start_dim=1)
        return x
    
    def decode(self, x):
        """Decode representation back to input space."""
        x = self.dec_cnn_layer1(x)
        x = self.dec_cnn_layer2(x)
        x = self.dec_cnn_layer3(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def forward(self, x):
        """Forward pass through encoder and decoder."""
        x = self.encode(x, False)
        x = self.decode(x)
        return x
    
# Motor prediction network for action prediction
class Motor_predictor_fwd(nn.Module):
    def __init__(self, action_dim, repr_shape, game_action_type='discrete', game_name=''):
        super(Motor_predictor_fwd, self).__init__()

        self.game_action_type = game_action_type
        self.n_neurons = 48
        self.feature_dim_d = repr_shape[1]
        self.act = nn.GELU()

        # Convolutional layers
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(self.feature_dim_d, 16, kernel_size=1, stride=1, padding=0),
            self.act
        )
        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0),
            self.act
        )

        # Dense layers
        self.layer_dense1 = nn.Linear(8 * 21 * 21, self.n_neurons)
        self.dropout1 = nn.Dropout(p=0.5)
        self.layer_dense2 = nn.Linear(self.n_neurons + 1, self.n_neurons)
        self.dropout2 = nn.Dropout(p=0.5)
        self.layer_out = nn.Linear(self.n_neurons, action_dim)
        
        self.init_weights()

        # Game-specific spatial masking
        if game_name == 'Enduro':
            self.uppercut, self.lowercut = 0, 6
        elif game_name == 'Freeway':
            self.uppercut, self.lowercut = 2, 0
        elif game_name == 'MsPacman':
            self.uppercut, self.lowercut = 0, 4
        elif game_name == 'Riverraid':
            self.uppercut, self.lowercut = 0, 5
        elif game_name == 'Seaquest':
            self.uppercut, self.lowercut = 5, 5
        elif game_name == 'SpaceInvaders':
            self.uppercut, self.lowercut = 4, 1
        else:
            raise ValueError(f"Unsupported game name: {game_name}")
        
    def init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, repr, lam):
        """Forward pass with spatial masking and lambda conditioning."""
        repr = repr.view(repr.shape[0], self.feature_dim_d, 21, 21)
        
        # Apply spatial masking
        if self.uppercut > 0:
            repr[:, :, :self.uppercut, :] *= 0
        if self.lowercut > 0:
            repr[:, :, -self.lowercut:, :] *= 0
        
        z = self.cnn_layer1(repr)
        z = self.cnn_layer2(z).flatten(start_dim=1)
        
        z = self.layer_dense1(z)
        z = self.act(z)
        z = self.dropout1(z)

        z = torch.cat([z, lam], dim=1)
        z = self.layer_dense2(z)
        z = self.act(z)
        z = self.dropout2(z)

        z = self.layer_out(z)

        if self.game_action_type == 'continuous':
            z[:, :-1] = F.tanh(z[:, :-1])
        return z


# Dilated convolution-based attention network
class CTR_Attention_dil(nn.Module):
    def __init__(self, repr_shape, config={}, autoencoder=[], device='auto', 
                 pos_enc_style='coord', enc_style='AE'):
        super().__init__()

        self.pos_enc_style = pos_enc_style
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.device = device
        self.repr_shape = repr_shape
        self.repr_size = np.prod(repr_shape)
        self.global_context_dim = 8
        self.autoencoder = autoencoder
        self.autoencoder.eval()
        self.enc_style = enc_style
        self.config = config

        self.psi_blend_dropout = config.get("psi_blend_dropout", 0.0)
        self.n_channels = 32

        if enc_style == 'self':
            self.feature_dim_d = self.n_channels
        elif enc_style == 'AE':
            self.feature_dim_d = repr_shape[1]
        
        self.feature_dim_x = self.repr_shape[-2]
        self.feature_dim_y = self.repr_shape[-1]

        # Coordinate encoding
        if self.pos_enc_style == 'coord':
            x = torch.linspace(-1, 1, self.feature_dim_x, device=self.device)
            y = torch.linspace(-1, 1, self.feature_dim_y, device=self.device)
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            self.coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
            self.coord_dims = 2

        # Projection layer
        self.pre_attn_proj = nn.Sequential(
            nn.Conv2d(self.feature_dim_d, self.n_channels, kernel_size=1)
        )

        # Dilated convolution layers
        self.attn_conv0 = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3, 
                                   stride=1, padding=1, dilation=1, padding_mode='reflect')
        self.attn_conv1 = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3, 
                                   stride=1, padding=3, dilation=3, padding_mode='reflect')
        self.attn_conv2 = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3, 
                                   stride=1, padding=5, dilation=5, padding_mode='reflect')
        self.attn_conv3 = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3, 
                                   stride=1, padding=7, dilation=7, padding_mode='reflect')
        
        # Global context
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.global_reduce = nn.Conv2d(self.feature_dim_d * 2, self.global_context_dim, kernel_size=1)

        # Attention predictor
        self.psi_predictor = nn.Sequential(
            nn.Linear(self.n_channels + self.global_context_dim + self.coord_dims + 1, 32),
            self.act,
            nn.Linear(32, 32),
            self.act,
            nn.Linear(32, 1)
        )
        
        self.apply(self._weights_init_kaiming)


    def _weights_init_kaiming(self, m):
        """Initialize weights using Kaiming initialization."""
        # Skip autoencoder submodules
        if hasattr(self, 'autoencoder') and any(m is mod for mod in self.autoencoder.modules()):
            return
    
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        
    def train(self, mode: bool = True):
        """Set training mode, keeping autoencoder in eval mode."""
        super().train(mode)
        self.autoencoder.eval()
        return self
    
    def loglam_to_lam(self, loglam):
        """Convert log lambda to lambda values."""
        range_ = 1
        lam = torch.where(
            loglam >= 0,
            10 ** (2 * range_ * loglam - range_),
            -10 ** (2 * range_ * (-loglam) - range_)
        )
        return lam

    def psi(self, state, lam, return_repr=False):
        """Compute attention weights psi."""
        batch_size = state.shape[0]

        with torch.no_grad():
            x_enc_raw = self.autoencoder.encode(state, flatten=False)

        x_enc = self.pre_attn_proj(x_enc_raw)

        # Global context
        global_avg = self.global_avg_pool(x_enc)
        global_max = self.global_max_pool(x_enc)
        global_context = torch.cat([global_avg, global_max], dim=1)
        global_context = self.global_reduce(global_context)
        global_context = global_context.expand(-1, -1, self.feature_dim_x, self.feature_dim_y)

        # Dilated convolutions
        x = self.attn_conv0(x_enc)
        x = self.act(x)
        x = self.attn_conv1(x)
        x = self.act(x)
        x = self.attn_conv2(x)
        x = self.act(x)
        x = self.attn_conv3(x)

        # Combine features
        lam = lam.unsqueeze(1).unsqueeze(1).repeat(1, 1, self.feature_dim_x, self.feature_dim_y)
        coord_channels = self.coords.repeat(batch_size, 1, 1, 1)
        x = torch.cat([x, global_context, coord_channels, lam], dim=1)

        # Predict attention
        x = x.permute(0, 2, 3, 1)
        x = self.psi_predictor(x)
        x = x.permute(0, 3, 1, 2)
        psi = torch.sigmoid(x)

        if return_repr:
            return psi, x_enc_raw
        else:
            return psi
        
    def forward(self, state, lam):
        """Forward pass with attention weighting."""
        psi, repr = self.psi(state, lam, return_repr=True)
        if self.training:
            psi = dropout_no_scaling(psi, p=self.psi_blend_dropout)
        return psi * repr
    
    def psi_overlayed_repr_fwd(self, state, repr, state_source, repr_source, lam):
        """Forward pass with overlayed representations."""
        psi = self.psi(state, lam)
        psi_raw = psi
        psi = dropout_no_scaling(psi, p=self.psi_blend_dropout)
        
        psi_source = self.psi(state_source, lam)
        repr_psi = psi * repr + torch.sqrt((1 - psi) * (1 - psi_source)).detach() * repr_source
        
        return repr_psi, psi_raw, psi_source
    
# Gaze prediction network using dilated convolutions
class Gaze_predictor_pool(nn.Module):
    def __init__(self,device, autoencoder):
        super().__init__()
        self.device=device
        self.autoencoder= autoencoder
        self.autoencoder=autoencoder.eval()
        self.repr_shape = self.autoencoder.give_repr_size()[1]
        
        self.feature_dim_d=self.repr_shape[1]
        self.feature_dim_x=self.repr_shape[-2]
        self.feature_dim_y=self.repr_shape[-1]

        self.n_channels=32
        self.global_context_dim=8
        self.act = nn.LeakyReLU(negative_slope=0.01)

        # Coord encoding
        x = torch.linspace(-1, 1, self.feature_dim_x,device=self.device)
        y = torch.linspace(-1, 1, self.feature_dim_y,device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        self.coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        self.coord_dims=2


        self.attn_conv0 = nn.Conv2d(self.feature_dim_d, self.n_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        self.attn_batch_norm0 = nn.BatchNorm2d(self.n_channels,momentum=0.01)
        
        self.attn_conv1 = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3, stride=1, padding=3, dilation=3)
        self.attn_batch_norm1 = nn.BatchNorm2d(self.n_channels,momentum=0.01)
        
        self.attn_conv2 = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3, stride=1, padding=5, dilation=5)
        self.attn_batch_norm2 = nn.BatchNorm2d(self.n_channels,momentum=0.01)
        
        self.attn_conv3 = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3, stride=1, padding=7, dilation=7)
        self.attn_batch_norm3 = nn.BatchNorm2d(self.n_channels,momentum=0.01)
        
        self.attn_conv4 = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3, stride=1, padding=9, dilation=9)
        self.attn_batch_norm4 = nn.BatchNorm2d(self.n_channels,momentum=0.01)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.global_reduce = nn.Conv2d(self.feature_dim_d*2, self.global_context_dim, kernel_size=1)
       
        self.psi_predictor = nn.Sequential(
            nn.Linear(self.n_channels+self.global_context_dim+self.coord_dims, 32),
            self.act,
            nn.Linear(32, 32),
            self.act,
            nn.Linear(32, 1)
        )
        
    def train(self, mode: bool = True):
        """Set training mode, keeping autoencoder in eval mode."""
        super().train(mode)
        self.autoencoder.eval()
        return self

    def forward(self, state):
        """Predict gaze distribution over spatial locations."""
     
        batch_size=state.shape[0]

        with torch.no_grad():
            x_enc=self.autoencoder.encode(state,flatten=False)
            x=x_enc
        
        global_avg = self.act(self.global_avg_pool(x_enc))
        global_max = self.act(self.global_max_pool(x_enc))
        global_context = torch.cat([global_avg,global_max],dim=1)
        global_context = self.global_reduce(global_context)
        global_context = global_context.expand(-1, -1, self.feature_dim_x, self.feature_dim_y)
        
        x=self.attn_conv0(x)
        x=self.act(x)
        x=self.attn_conv1(x)
        x=self.act(x)
        x=self.attn_conv2(x)
        x=self.act(x)
        x=self.attn_conv3(x)
        x=self.act(x)
        x=self.attn_conv4(x)
        x=self.act(x)
        
        
        coord_channels = self.coords.expand(batch_size, -1, -1, -1)
        x = torch.cat([x, global_context, coord_channels], dim=1)

        x = x.permute(0,2,3,1)
        x = self.psi_predictor(x)
        x = x.permute(0,3,1,2)

        x=x.flatten(start_dim=1)

        x=torch.softmax(x,dim=1)

        x=x.view(batch_size,self.feature_dim_x,self.feature_dim_y) # (batch_size,26,20,1)

        return x
