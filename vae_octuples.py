import torch
import torch.nn as nn
import torch.nn.functional as F

class OctupleVAE(nn.Module):
    def __init__(self, vocab_sizes, embed_dim=64, hidden_dim=256, latent_dim=128, num_layers=1, device='cuda'):
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

        # 8 separate embeddings for each channel
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim) for vocab_size in vocab_sizes
        ])

        # encoder GRU
        self.encoder_gru = nn.GRU(embed_dim * 8, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        # latent vectors used for reparametrization
        self.fc_mean = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        # eecoder initial hidden state from z
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)

        # encoder GRU
        self.decoder_gru = nn.GRU(embed_dim * 8, hidden_dim, num_layers, batch_first=True)

        # output projections for each channel to turn hidden states into vocab logits
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
        
        # find latent vectors
        z_mean = self.fc_mean(h_last)
        z_logvar = self.fc_logvar(h_last)
        return z_mean, z_logvar

    def encode_with_sequence(self, x):
        """Return full encoder sequence along with latent parameters.
        x: [batch, seq_len, 8]
        Returns:
            h_enc: [batch, seq_len, hidden_dim*2]
            z_mean: [batch, latent_dim]
            z_logvar: [batch, latent_dim]
        """
        embeds = [self.embeddings[i](x[:, :, i]) for i in range(8)]
        x_emb = torch.cat(embeds, dim=-1)
        h_enc, _ = self.encoder_gru(x_emb)
        h_last = h_enc[:, -1, :]
        z_mean = self.fc_mean(h_last)
        z_logvar = self.fc_logvar(h_last)
        return h_enc, z_mean, z_logvar

    def reparameterize(self, mu, logvar):
        """Returns z sample from latent distribution"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len, x=None, autoregressive=False, x_prefix=None, temperature=1.0, top_k=None, start_tokens=None):
        """
        z: [batch, latent_dim]
        seq_len: length of output sequence
        x: optional teacher-forcing input [batch, seq_len, 8]
        autoregressive: if True and x is None, use own predictions as input
        x_prefix: optional teacher-forced prefix [batch, t0, 8] to warm-up
        temperature: sampling temperature for autoregressive decoding
        top_k: if set, sample from top-k logits
        start_tokens: optional initial tokens [batch, 8] for t=0
        """
        batch_size = z.size(0)
        hidden = self.fc_hidden(z)
        hidden = hidden.view(self.decoder_gru.num_layers, batch_size, self.hidden_dim)

        if x is not None:
            # Teacher forcing mode (training)
            embeds = [self.embeddings[i](x[:, :, i]) for i in range(8)]
            x_emb = torch.cat(embeds, dim=-1)
            h_dec, _ = self.decoder_gru(x_emb, hidden)
            outputs = [layer(h_dec) for layer in self.output_layers]
            return outputs
        
        elif autoregressive:
            # Autoregressive mode: use own predictions
            outputs = [[] for _ in range(8)]
            # Helper: sample from logits
            def sample_logits(logits):
                if top_k is not None and top_k > 0:
                    topk_vals, topk_idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                    probs = F.softmax(topk_vals / max(temperature, 1e-6), dim=-1)
                    idx = torch.multinomial(probs, num_samples=1)
                    return torch.gather(topk_idx, -1, idx).squeeze(-1)
                else:
                    probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
                    return torch.multinomial(probs, num_samples=1).squeeze(-1)

            # If a teacher-forced prefix is provided, run it first to set hidden state
            if x_prefix is not None and x_prefix.size(1) > 0:
                embeds_pf = [self.embeddings[i](x_prefix[:, :, i]) for i in range(8)]
                x_pf = torch.cat(embeds_pf, dim=-1)
                _, hidden = self.decoder_gru(x_pf, hidden)

            # Start tokens: provided or zeros
            if start_tokens is not None:
                x_t = start_tokens.to(self.device)
            else:
                x_t = torch.zeros(batch_size, 8, dtype=torch.long, device=self.device)
            
            for t in range(seq_len):
                # Embed current input
                embeds = [self.embeddings[i](x_t[:, i]) for i in range(8)]
                x_emb = torch.cat(embeds, dim=-1).unsqueeze(1)  # [batch, 1, embed_dim*8]
                
                # Decode one step
                h_dec, hidden = self.decoder_gru(x_emb, hidden)
                h_dec = h_dec.squeeze(1)  # [batch, hidden_dim]
                
                # Predict next tokens
                preds = []
                for i, layer in enumerate(self.output_layers):
                    logits = layer(h_dec)  # [batch, vocab_i]
                    outputs[i].append(logits)
                    # Sample next token
                    pred = sample_logits(logits)
                    preds.append(pred)
                
                # Use predictions as next input
                x_t = torch.stack(preds, dim=-1)  # [batch, 8]
            
            # Stack timesteps: [batch, seq_len, vocab_i]
            outputs = [torch.stack(out, dim=1) for out in outputs]
            return outputs
        
        else:
            # Old behavior: zeros (will produce garbage)
            x_input = torch.zeros(batch_size, seq_len, 8, dtype=torch.long, device=self.device)
            embeds = [self.embeddings[i](x_input[:, :, i]) for i in range(8)]
            x_emb = torch.cat(embeds, dim=-1)
            h_dec, _ = self.decoder_gru(x_emb, hidden)
            outputs = [layer(h_dec) for layer in self.output_layers]
            return outputs

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        outputs = self.decode(z, seq_len=x.size(1), x=x)
        return outputs, mu, logvar

# Loss function
def vae_loss(outputs, x, mu, logvar, KL_weight=1.0):
    """
    outputs: list of 8 tensors [batch, seq_len, vocab_i]
    x: [batch, seq_len, 8] long
    """
    # reconstruction loss
    recon_loss = 0
    for i in range(8):
        recon_loss += F.cross_entropy(outputs[i].permute(0, 2, 1), x[:, :, i], reduction='mean')
    
    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + KL_weight * kl_loss, recon_loss, kl_loss
