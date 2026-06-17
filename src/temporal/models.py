"""
src/temporal/models.py — All 4 spatiotemporal prediction models (Table 3 in paper)

    Model            Architecture                           Loss
    
    ConvLSTM         Single ConvLSTM (32ch) + Conv decoder  L1
    PredRNNPP        4-layer ST-LSTM + GHU                  L1
    MetadataFusion   CNN enc-dec + FiLM conditioning         MSE+SSIM
    PhyDNet          PhyCell + residual ConvLSTM             L1

All models:
  - Input:  2 consecutive context frames (B, T=2, 1, H, W) in [0, 1]
  - Output: predicted next frame (B, 1, H, W) in [0, 1]
  - Image size: 128 × 128

Usage:
    from src.temporal.models import ConvLSTM, PredRNNPP, MetadataFusion, PhyDNet
    model = ConvLSTM()
    pred  = model(ctx_frames)   # ctx_frames: (B, T, 1, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# (1/4) ConvLSTM 

class ConvLSTMCell(nn.Module):
    """Standard convolutional LSTM cell (Shi et al. 2015)."""
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        p = kernel_size // 2
        self.conv       = nn.Conv2d(input_dim + hidden_dim,
                                    4 * hidden_dim, kernel_size, padding=p)
        self.hidden_dim = hidden_dim

    def forward(self, x, h, c):
        i, f, o, g = torch.chunk(self.conv(torch.cat([x, h], 1)), 4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c

    def init_hidden(self, B: int, H: int, W: int, device):
        z = torch.zeros(B, self.hidden_dim, H, W, device=device)
        return z, z.clone()


class ConvLSTM(nn.Module):
    """
    Single ConvLSTM cell (hidden_dim=32) + 1×1 Conv decoder.
    Processes the sequence frame-by-frame; hidden state after the
    last frame is decoded into the predicted next frame.
    """
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32):
        super().__init__()
        self.cell    = ConvLSTMCell(input_dim, hidden_dim)
        self.decoder = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x_seq.shape
        h, c = self.cell.init_hidden(B, H, W, x_seq.device)
        for t in range(T):
            h, c = self.cell(x_seq[:, t], h, c)
        return self.decoder(h)


# (2/4) PredRNN++

class SpatioTemporalLSTMCell(nn.Module):
    """
    ST-LSTM cell from PredRNN++ (Wang et al. 2018).
    Maintains two memory states: H (spatiotemporal zigzag) and M (spatial).
    """
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        p  = kernel_size // 2
        id = input_dim + hidden_dim
        self.hidden_dim = hidden_dim
        self.conv_h = nn.Conv2d(id, 4 * hidden_dim, kernel_size, padding=p)
        self.conv_m = nn.Conv2d(id, 3 * hidden_dim, kernel_size, padding=p)
        self.conv_o = nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size, padding=p)

    def forward(self, x, h, c, m):
        xh = torch.cat([x, h], 1)
        i, g, f, _ = torch.chunk(self.conv_h(xh), 4, 1)
        i = torch.sigmoid(i); f = torch.sigmoid(f); g = torch.tanh(g)
        c_new = f * c + i * g

        xm = torch.cat([x, m], 1)
        i2, f2, g2 = torch.chunk(self.conv_m(xm), 3, 1)
        i2 = torch.sigmoid(i2); f2 = torch.sigmoid(f2); g2 = torch.tanh(g2)
        m_new = f2 * m + i2 * g2

        o  = torch.sigmoid(self.conv_o(torch.cat([c_new, m_new, _], 1)))
        # combine c and m memory via 1×1 conv
        w  = torch.eye(self.hidden_dim, 2 * self.hidden_dim,
                        device=x.device).view(self.hidden_dim, 2 * self.hidden_dim, 1, 1)
        h_new = o * torch.tanh(F.conv2d(torch.cat([c_new, m_new], 1), w))
        return h_new, c_new, m_new

    def init_hidden(self, B, H, W, device):
        z = torch.zeros(B, self.hidden_dim, H, W, device=device)
        return z, z.clone(), z.clone()


class GradientHighwayUnit(nn.Module):
    """GHU bridging layers 0→1 in PredRNN++."""
    def __init__(self, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        p = kernel_size // 2
        self.conv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2,
                              kernel_size, padding=p)

    def forward(self, x, z):
        p, u = torch.chunk(self.conv(torch.cat([x, z], 1)), 2, 1)
        u = torch.sigmoid(u)
        return u * torch.tanh(p) + (1 - u) * z


class PredRNNPP(nn.Module):
    """
    PredRNN++ with 4 ST-LSTM layers and a Gradient Highway Unit.
    Channel dims: [32, 64, 64, 64] matching Table 3 in the paper.
    Best temporal model for cross-dataset generalisation (Table 9).
    """
    def __init__(self, input_dim: int = 1,
                 dims: tuple = (32, 64, 64, 64)):
        super().__init__()
        self.dims  = dims
        self.cells = nn.ModuleList()
        in_d = input_dim
        for d in dims:
            self.cells.append(SpatioTemporalLSTMCell(in_d, d))
            in_d = d
        self.ghu     = GradientHighwayUnit(dims[0])
        self.decoder = nn.Conv2d(dims[-1], 1, 1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x_seq.shape
        dev    = x_seq.device
        states = [c.init_hidden(B, H, W, dev) for c in self.cells]
        z      = torch.zeros(B, self.dims[0], H, W, device=dev)

        for t in range(T):
            x = x_seq[:, t]
            h0, c0, m0 = self.cells[0](x, *states[0])
            z           = self.ghu(h0, z)
            states[0]   = (h0, c0, m0)
            inp = z
            for li in range(1, len(self.cells)):
                h, c, m   = self.cells[li](inp, *states[li])
                states[li] = (h, c, m)
                inp        = h

        return self.decoder(states[-1][0])


# (3/4) MetadataFusion

def _cb(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class MetadataFusion(nn.Module):
    """
    CNN encoder-decoder with FiLM conditioning from protocol metadata.

    Categorical metadata (microscope, cell_line, culture_medium,
    formation_method, magnification) are embedded and combined with
    continuous features (seeding_density_num, Δt) to generate per-channel
    scale (γ) and shift (β) at the encoder bottleneck.

    Args:
        seq_len:            number of input context frames (default 2)
        base_ch:            base channel width (default 32)
        cat_cardinalities:  {cat_col: n_categories}
        emb_dim:            embedding dimension for categorical features (default 32)
        cont_dim:           number of continuous features (default 2)
    """
    def __init__(self, seq_len: int = 2, base_ch: int = 32,
                 cat_cardinalities: dict = None,
                 emb_dim: int = 32, cont_dim: int = 2):
        super().__init__()
        B = base_ch

        # Encoder
        self.e1 = _cb(seq_len, B);  self.p1 = nn.MaxPool2d(2)
        self.e2 = _cb(B,   B*2);    self.p2 = nn.MaxPool2d(2)
        self.e3 = _cb(B*2, B*4);    self.p3 = nn.MaxPool2d(2)
        self.e4 = _cb(B*4, B*8);    self.p4 = nn.MaxPool2d(2)
        bot_ch  = B * 8

        # Metadata + FiLM
        cat_cardinalities = cat_cardinalities or {}
        self.embs   = nn.ModuleDict({
            c: nn.Embedding(n, emb_dim)
            for c, n in cat_cardinalities.items()
        })
        self.cat_cols = list(cat_cardinalities.keys())
        meta_in       = len(self.cat_cols) * emb_dim + cont_dim
        self.meta_mlp = nn.Sequential(nn.Linear(meta_in, 256), nn.ReLU(),
                                       nn.Linear(256, 256), nn.ReLU())
        self.film_gen = nn.Linear(256, 2 * bot_ch)
        self._bot_ch  = bot_ch

        # Decoder with skip connections
        self.u4 = nn.ConvTranspose2d(bot_ch, B*4, 2, 2); self.d4 = _cb(B*4+B*8, B*4)
        self.u3 = nn.ConvTranspose2d(B*4,   B*2, 2, 2); self.d3 = _cb(B*2+B*4, B*2)
        self.u2 = nn.ConvTranspose2d(B*2,   B,   2, 2); self.d2 = _cb(B  +B*2, B)
        self.u1 = nn.ConvTranspose2d(B,     B,   2, 2); self.d1 = _cb(B  +B,   B)
        self.final = nn.Sequential(nn.Conv2d(B, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x_seq: torch.Tensor,
                cat_codes: torch.Tensor,
                cont: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq:     (B, T, H, W) context frames (T channels, no C dim)
            cat_codes: (B, n_cat)  integer category indices
            cont:      (B, 2)      [seeding_density_num, delta_t]
        """
        e1 = self.e1(x_seq);  p1 = self.p1(e1)
        e2 = self.e2(p1);     p2 = self.p2(e2)
        e3 = self.e3(p2);     p3 = self.p3(e3)
        e4 = self.e4(p3);     b  = self.p4(e4)

        # FiLM
        embs = torch.cat([self.embs[c](cat_codes[:, i])
                          for i, c in enumerate(self.cat_cols)], dim=1)
        m    = self.meta_mlp(torch.cat([embs, cont], dim=1))
        gam, bet = torch.chunk(self.film_gen(m), 2, dim=1)
        b = b * (1 + gam.view(-1, self._bot_ch, 1, 1)) \
            + bet.view(-1, self._bot_ch, 1, 1)

        d = self.d4(torch.cat([self.u4(b), e4], 1))
        d = self.d3(torch.cat([self.u3(d), e3], 1))
        d = self.d2(torch.cat([self.u2(d), e2], 1))
        d = self.d1(torch.cat([self.u1(d), e1], 1))
        return self.final(d)


# (4/4) PhyDNet

class PhyCell(nn.Module):
    """
    Physics-guided recurrent cell.
    Learns a bank of spatial difference operators and integrates
    dh/dt = F(h) via an Euler step with learnable coefficients.
    """
    def __init__(self, channels: int, num_ops: int = 3,
                 kernel_size: int = 3):
        super().__init__()
        p = kernel_size // 2
        self.ops    = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size,
                      padding=p, groups=channels, bias=False)
            for _ in range(num_ops)])
        self.coeffs = nn.Parameter(
            torch.randn(num_ops, channels, 1, 1) * 0.01)
        self.gate   = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor,
                dt=1.0) -> torch.Tensor:
        ops_out = sum(self.coeffs[k] * op(h_prev)
                      for k, op in enumerate(self.ops))
        du      = torch.tanh(ops_out)
        if isinstance(dt, torch.Tensor):
            dt = dt.view(-1, 1, 1, 1)
        h_tilde = h_prev + dt * du
        g       = torch.sigmoid(self.gate(torch.cat([x, h_tilde], 1)))
        return g * h_tilde + (1 - g) * h_prev


class PhyDNet(nn.Module):
    """
    PhyCell (physics dynamics) + residual ConvLSTM (residual dynamics).
    Physics-inspired separation of interpretable and data-driven components.
    Improves structural prediction quality over vanilla ConvLSTM.

    Args:
        enc_ch:  encoder output channels (default 64)
        phy_ch:  PhyCell hidden channels (default 32)
        res_ch:  residual ConvLSTM hidden channels (default 32)
    """
    def __init__(self, enc_ch: int = 64, phy_ch: int = 32,
                 res_ch: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, enc_ch // 2, 3, padding=1),
            nn.BatchNorm2d(enc_ch // 2), nn.ReLU(inplace=True),
            nn.Conv2d(enc_ch // 2, enc_ch, 3, padding=1),
            nn.BatchNorm2d(enc_ch), nn.ReLU(inplace=True),
            nn.Conv2d(enc_ch, phy_ch + res_ch, 1))
        self.phy_ch  = phy_ch
        self.res_ch  = res_ch
        self.phycell = PhyCell(phy_ch)
        self.rescell = ConvLSTMCell(res_ch, res_ch)
        self.decoder = nn.Sequential(
            nn.Conv2d(phy_ch + res_ch, enc_ch, 3, padding=1),
            nn.BatchNorm2d(enc_ch), nn.ReLU(inplace=True),
            nn.Conv2d(enc_ch, enc_ch // 2, 3, padding=1),
            nn.BatchNorm2d(enc_ch // 2), nn.ReLU(inplace=True),
            nn.Conv2d(enc_ch // 2, 1, 1), nn.Sigmoid())

    def forward(self, x_seq: torch.Tensor,
                dt: torch.Tensor = None) -> torch.Tensor:
        B, T, C, H, W = x_seq.shape
        dev   = x_seq.device
        phy_h = torch.zeros(B, self.phy_ch, H, W, device=dev)
        res_h, res_c = self.rescell.init_hidden(B, H, W, dev)

        for t in range(T):
            z     = self.encoder(x_seq[:, t])
            step  = dt if (t == T - 1 and dt is not None) else 1.0
            phy_h = self.phycell(z[:, :self.phy_ch], phy_h, dt=step)
            res_h, res_c = self.rescell(z[:, self.phy_ch:], res_h, res_c)

        return self.decoder(torch.cat([phy_h, res_h], 1))