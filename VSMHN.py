import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal, kl_divergence
from tqdm.auto import tqdm


class Encoder_TS(nn.Module):
    def __init__(self, x_dim, h_dim, phi_x, phi_tf, use_GRU=True):
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.phi_x = phi_x
        self.phi_tf = phi_tf
        self.use_GRU = use_GRU
        if (use_GRU):
            self.rnn = nn.GRU(2*h_dim, h_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(2*h_dim, h_dim, batch_first=True)

    def forward(self, src, tf):
        # src : (batch_size, seq_len, x_dim)
        x_h = self.phi_x(src)
        tf_h = self.phi_tf(tf)
        joint_h = torch.cat([x_h, tf_h], -1)
        if (self.use_GRU):
            outputs, hidden = self.rnn(joint_h)
        else:
            outputs, (hidden, cell_state) = self.rnn(joint_h)
        return hidden


class Encoder_Event(nn.Module):
    def __init__(self, x_dim, h_dim, bound=0.05, use_GRU=True):
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.use_GRU = use_GRU
        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim))
        if (use_GRU):
            self.rnn = nn.GRU(h_dim, h_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(h_dim, h_dim, batch_first=True)

    def forward(self, src):
        x_h = self.phi_x(src)
        if (self.use_GRU):
            outputs, hidden = self.rnn(x_h)
        else:
            outputs, (hidden, cell_state) = self.rnn(x_h)
        return hidden


class Hidden_Encoder(nn.Module):
    def __init__(self, h_dim, z_dim):
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.enc_mean = nn.Sequential(
            nn.Linear(h_dim, z_dim))
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

    def forward(self, h):
        return Normal(self.enc_mean(h), self.enc_std(h))


class Hidden_Decoder(nn.Module):
    def __init__(self, h_dim, z_dim):
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.dec = nn.Sequential(
            nn.Linear(z_dim, h_dim))

    def forward(self, z):
        return self.dec(z)


class Decoder_TS(nn.Module):
    def __init__(self, x_dim, h_dim, phi_x, phi_tf, bound=0.05, use_GRU=True):
        super().__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.phi_x = phi_x
        self.phi_tf = phi_tf
        self.use_GRU = use_GRU
        if (use_GRU):
            self.rnn = nn.GRUCell(2*h_dim, 2*h_dim)
        else:
            self.rnn = nn.LSTMCell(2*h_dim, 2*h_dim)
        self.dec_mean = nn.Sequential(
            nn.Linear(2*h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim))
        self.dec_std = nn.Sequential(
            nn.Linear(2*h_dim, x_dim),
            nn.Softplus())
        self.bound = bound

    def forward(self, x_t, tf_t, hidden):
        x_h = self.phi_x(x_t)
        tf_h = self.phi_tf(tf_t)
        joint_h = torch.cat([x_h, tf_h], -1)
        if (self.use_GRU):
            hidden = self.rnn(joint_h, hidden)
        else:
            (hidden, cell_state) = self.rnn(joint_h, hidden)
        x_mu = self.dec_mean(hidden)
        x_std = self.dec_std(hidden) + x_t.new_tensor([self.bound])
        if (self.use_GRU):
            return Normal(x_mu, x_std), hidden
        else:
            return Normal(x_mu, x_std), (hidden, cell_state)


class VSMHN(nn.Module):
    def __init__(self, device, ts_dim, event_dim, tf_dim, h_dim, z_dim, forecast_horizon, dec_bound=0.1, use_GRU=True):
        super().__init__()
        self.device = device
        self.ts_dim = ts_dim
        self.event_dim = event_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.forecast_horizon = forecast_horizon
        self.use_GRU = use_GRU
        self.phi_ts = nn.Sequential(nn.Linear(ts_dim, h_dim),
                                    nn.ReLU(),
                                    nn.Linear(h_dim, h_dim))
        self.phi_tf = nn.Sequential(nn.Linear(tf_dim, h_dim),
                                    nn.ReLU(),
                                    nn.Linear(h_dim, h_dim))
        self.ts_encoder = Encoder_TS(ts_dim, h_dim, self.phi_ts, self.phi_tf, self.use_GRU)
        self.event_encoder = Encoder_Event(event_dim, h_dim, self.use_GRU)
        self.ts_decoder = Decoder_TS(ts_dim, h_dim, self.phi_ts,
                                     self.phi_tf, bound=dec_bound, use_GRU=self.use_GRU)
        self.posterior_encoder = Hidden_Encoder(2*h_dim, z_dim)
        self.prior_encoder = Hidden_Encoder(2*h_dim, z_dim)
        self.hidden_decoder = Hidden_Decoder(h_dim, z_dim)
        self.temporal_decay = np.linspace(1.5, 0.5, forecast_horizon)
        self.phi_dec = nn.Sequential(nn.Linear(3*h_dim, 2*h_dim))

    def forward(self, ts_past, event_past, ts_tf_past, ts_trg, tf_future):
        # seq : shape [batch_size, seq_len, x_dim]
        # trg : shape [batch_size, seq_len, x_dim]
        # tf_fugure shape [batch_size, seq_len, tf_dim]
        # event_past: shape [batch_size, seq_len, event_dim]
        ts_hidden = self.ts_encoder(ts_past, ts_tf_past).squeeze(0)
        ts_hidden_tau = self.ts_encoder(torch.cat([ts_past, ts_trg], dim=1),
                                        torch.cat([ts_tf_past, tf_future], dim=1)).squeeze(0)
        event_hidden = self.event_encoder(event_past).squeeze(0)
        joint_hidden = torch.cat([ts_hidden, event_hidden], dim=-1)
        joint_hidden_tau = torch.cat([ts_hidden_tau, event_hidden], dim=-1)
        pz_rv = self.prior_encoder(joint_hidden)
        qz_rv = self.posterior_encoder(joint_hidden_tau)
        z = qz_rv.rsample()
        z_dec = self.hidden_decoder(z)
        hidden_dec = self.phi_dec(torch.cat([joint_hidden, z_dec], dim=-1))
        if not self.use_GRU:
            hidden_dec = (hidden_dec, torch.zeros(hidden_dec.shape).to(self.device))
        ts_t = ts_past[:, -1, :]
        likelihoods = []
        for t in range(self.forecast_horizon):
            tf_t = tf_future[:, t, :]
            ts_t_rv, hidden_dec = self.ts_decoder(ts_t, tf_t, hidden_dec)
            likelihoods.append(ts_t_rv.log_prob(ts_trg[:, t, :]))
            ts_t = ts_t_rv.sample()
        likelihoods = torch.stack(likelihoods, dim=1)
        kls = kl_divergence(qz_rv, pz_rv)
        return torch.mean(torch.sum(likelihoods, (-1, -2))), kls.mean()


def predict(model, ts_past, event_past, ts_tf_past, tf_future, mc_times=100):
    # seq : shape [batch_size, seq_len, x_dim]
    # trg : shape [batch_size, seq_len, x_dim]
    # tf_fugure shape [batch_size, seq_len, tf_dim]
    # event_past: shape [batch_size, seq_len, event_dim]
    ts_hidden = model.ts_encoder(ts_past, ts_tf_past).squeeze(0)
    event_hidden = model.event_encoder(event_past).squeeze(0)
    joint_hidden = torch.cat([ts_hidden, event_hidden], dim=-1)
    pz_rv = model.prior_encoder(joint_hidden)
    predictions = np.zeros(
        shape=(mc_times, tf_future.shape[0], tf_future.shape[1], tf_future.shape[2]))
    for idx in tqdm(range(mc_times), leave=True):
        z = pz_rv.sample()
        z_dec = model.hidden_decoder(z)
        hidden_dec = model.phi_dec(torch.cat([joint_hidden, z_dec], dim=-1))
        if (not model.use_GRU):
            hidden_dec = (hidden_dec, torch.zeros(hidden_dec.shape).to(model.device))
        ts_t = ts_past[:, -1, :]
        for t in range(model.forecast_horizon):
            tf_t = tf_future[:, t, :]
            ts_t_rv, hidden_dec = model.ts_decoder(ts_t, tf_t, hidden_dec)
            ts_t = ts_t_rv.sample()
            predictions[idx, :, t, :] = ts_t.cpu().numpy()
    return predictions
