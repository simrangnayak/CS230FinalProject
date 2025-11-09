import torch
import torch.nn as nn
import torch.nn.functional as F

class OctupleVAE_HierarchicalDecoder(nn.Module):
    def __init__(self, vocab_sizes, embed_dim=64, hidden_dim=256, latent_dim=128, num_layers=1, chunks=4, device='cuda'):
        """
        vocab_sizes : list of 8 ints, vocab size for each octuple channel
        embed_dim   : embedding dim for each channel
        hidden_dim  : GRU hidden dim
        latent_dim  : size of latent vector z
        """
        super().__init__()
        self.device = device
        self.vocab_sizes = vocab_sizes
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.chunks = chunks  # number of chunks for hierarchical decoding

        # 8 separate embeddings for each channel
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim) for vocab_size in vocab_sizes
        ])

        # Encoder GRU
        self.encoder_gru = nn.GRU(embed_dim * 8, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        # Latent vectors
        self.fc_mean = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        # Decoder initial hidden state from z
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)

        # Hierarchical Decoder
        self.high_rnn = nn.GRU(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.low_rnn = nn.GRU(embed_dim*8 + hidden_dim, hidden_dim, num_layers, batch_first=True)

        # Output projections for each channel
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size) for vocab_size in vocab_sizes
        ])

    def encode(self, x):
        """
        x: [batch, seq_len, 8] long tensor
        """
        # embed each channel
        embeds = [self.embeddings[i](x[:, :, i]) for i in range(8)]
        x_emb = torch.cat(embeds, dim=-1)  # [batch, seq_len, embed_dim*8]

        h_enc, _ = self.encoder_gru(x_emb)  # [batch, seq_len, hidden*2]
        h_last = h_enc[:, -1, :]            # take last timestep
        z_mean = self.fc_mean(h_last)
        z_logvar = self.fc_logvar(h_last)
        return z_mean, z_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len, x=None):
        """
        z: [batch, latent_dim]
        seq_len: length of output sequence
        x: optional teacher-forcing input [batch, seq_len, 8]
        """
        z_expanded = z.unsqueeze(1).repeat(1, self.chunks, 1)  # [batch, chunks, latent_dim]
        high_hidden, _ = self.high_rnn(z_expanded)  # [batch, chunks, hidden_dim]
        
        all_outputs = []
        batch_size = z.size(0)
        for i in range(self.chunks):
            chunk_hidden = high_hidden[:, i, :].unsqueeze(0).repeat(self.low_rnn.num_layers, 1, 1)  # [num_layers, batch, hidden_dim]
            if x is None:
                x_input = torch.zeros(batch_size, seq_len // self.chunks, 8, dtype=torch.long, device=self.device)
            else:
                x_input = x[:, i*(seq_len//self.chunks):(i+1)*(seq_len//self.chunks), :]

            embeds = [self.embeddings[j](x_input[:, :, j]) for j in range(8)]
            x_emb = torch.cat(embeds, dim=-1)  # [batch, chunk_seq_len, embed_dim*8]

            # Concatenate high-level hidden state to each timestep input
            high_context = chunk_hidden[-1].unsqueeze(1).repeat(1, x_emb.size(1), 1)  # [batch, chunk_seq_len, hidden_dim]
            low_input = torch.cat([x_emb, high_context], dim=-1)  # [batch, chunk_seq_len, embed_dim*8 + hidden_dim]

            h_dec, _ = self.low_rnn(low_input, chunk_hidden)
            outputs = [layer(h_dec) for layer in self.output_layers]  # list of [batch, chunk_seq_len, vocab_i]
            
            if i == 0:
                all_outputs = outputs
            else:
                for j in range(8):
                    all_outputs[j] = torch.cat([all_outputs[j], outputs[j]], dim=1)  # concatenate along seq_len

        return all_outputs

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        outputs = self.decode(z, seq_len=x.size(1), x=x)
        return outputs, mu, logvar

# Loss function
def vae_loss(outputs, x, mu, logvar):
    """
    outputs: list of 8 tensors [batch, seq_len, vocab_i]
    x: [batch, seq_len, 8] long
    """
    recon_loss = 0
    for i in range(8):
        recon_loss += F.cross_entropy(outputs[i].permute(0, 2, 1), x[:, :, i], reduction='mean')
    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss
