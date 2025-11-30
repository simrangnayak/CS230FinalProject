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

        # encoder GRU
        self.encoder_gru = nn.GRU(embed_dim * 8, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        # latent vectors used for reparametrization
        self.fc_mean = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        # decoder initial hidden state from z
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)

        # hierarchical Decoder
        self.high_rnn = nn.GRU(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.low_rnn = nn.GRU(embed_dim*8 + hidden_dim, hidden_dim, num_layers, batch_first=True)

        # dropout for additional regularization
        self.dropout = nn.Dropout(0.3)
        
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

    def reparameterize(self, mu, logvar):
        """Returns z sample from latent distribution"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len, x=None, autoregressive=False, x_prefix=None, temperature=1.0, top_k=None):
        """
        z: [batch, latent_dim]
        seq_len: length of output sequence
        x: optional teacher-forcing input [batch, seq_len, 8]
        autoregressive: if True and x is None, use own predictions
        x_prefix: optional teacher-forced prefix [batch, t0, 8] to warm-up
        temperature: sampling temperature for autoregressive decoding
        top_k: if set, sample from top-k logits
        """
        # process z through high-level RNN
        z_expanded = z.unsqueeze(1).repeat(1, self.chunks, 1)  # [batch, chunks, latent_dim]
        high_hidden, _ = self.high_rnn(z_expanded)  # [batch, chunks, hidden_dim]
        
        batch_size = z.size(0)
        chunk_len = seq_len // self.chunks
        
        if x is not None:
            # Teacher forcing mode
            all_outputs = []
            for i in range(self.chunks):
                chunk_hidden = high_hidden[:, i, :].unsqueeze(0).repeat(self.low_rnn.num_layers, 1, 1)
                x_input = x[:, i*chunk_len:(i+1)*chunk_len, :]
                embeds = [self.embeddings[j](x_input[:, :, j]) for j in range(8)]
                x_emb = torch.cat(embeds, dim=-1)
                high_context = chunk_hidden[-1].unsqueeze(1).repeat(1, x_emb.size(1), 1)
                low_input = torch.cat([x_emb, high_context], dim=-1)
                h_dec, _ = self.low_rnn(low_input, chunk_hidden)
                outputs = [layer(h_dec) for layer in self.output_layers]
                
                if i == 0:
                    all_outputs = outputs
                else:
                    for j in range(8):
                        all_outputs[j] = torch.cat([all_outputs[j], outputs[j]], dim=1)
            return all_outputs
        
        elif autoregressive:
            # Autoregressive mode
            def sample_logits(logits):
                if top_k is not None and top_k > 0:
                    topk_vals, topk_idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                    probs = F.softmax(topk_vals / max(temperature, 1e-6), dim=-1)
                    idx = torch.multinomial(probs, num_samples=1)
                    return torch.gather(topk_idx, -1, idx).squeeze(-1)
                else:
                    probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
                    return torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            all_outputs = [[] for _ in range(8)]
            last_tokens = None  # Keep track of last generated tokens
            
            for i in range(self.chunks):
                chunk_hidden = high_hidden[:, i, :].unsqueeze(0).repeat(self.low_rnn.num_layers, 1, 1)
                
                # Process prefix if provided and spans into this chunk
                chunk_start = i * chunk_len
                chunk_end = (i + 1) * chunk_len
                
                if x_prefix is not None and chunk_start < x_prefix.size(1):
                    # Use prefix tokens for this chunk if available
                    prefix_end_in_chunk = min(x_prefix.size(1) - chunk_start, chunk_len)
                    if prefix_end_in_chunk > 0:
                        chunk_prefix = x_prefix[:, chunk_start:chunk_start + prefix_end_in_chunk, :]
                        embeds_pf = [self.embeddings[j](chunk_prefix[:, :, j]) for j in range(8)]
                        x_pf = torch.cat(embeds_pf, dim=-1)
                        high_context_pf = chunk_hidden[-1].unsqueeze(1).repeat(1, prefix_end_in_chunk, 1)
                        low_input_pf = torch.cat([x_pf, high_context_pf], dim=-1)
                        h_dec_pf, chunk_hidden = self.low_rnn(low_input_pf, chunk_hidden)
                        
                        # Store prefix outputs and get last tokens
                        for j, layer in enumerate(self.output_layers):
                            prefix_logits = layer(h_dec_pf)
                            all_outputs[j].extend([prefix_logits[:, t, :] for t in range(prefix_end_in_chunk)])
                        
                        last_tokens = chunk_prefix[:, -1, :] if prefix_end_in_chunk > 0 else None
                        start_t = prefix_end_in_chunk
                    else:
                        start_t = 0
                else:
                    start_t = 0
                
                # Start with last tokens from previous chunk or zeros
                if last_tokens is not None:
                    x_t = last_tokens
                else:
                    x_t = torch.zeros(batch_size, 8, dtype=torch.long, device=self.device)
                
                # Generate remaining tokens in this chunk
                for t in range(start_t, chunk_len):
                    embeds = [self.embeddings[j](x_t[:, j]) for j in range(8)]
                    x_emb = torch.cat(embeds, dim=-1).unsqueeze(1)
                    high_context = chunk_hidden[-1].unsqueeze(1)
                    low_input = torch.cat([x_emb, high_context], dim=-1)
                    h_dec, chunk_hidden = self.low_rnn(low_input, chunk_hidden)
                    h_dec = h_dec.squeeze(1)
                    h_dec = self.dropout(h_dec)  # Apply dropout
                    
                    preds = []
                    for j, layer in enumerate(self.output_layers):
                        logits = layer(h_dec)
                        all_outputs[j].append(logits)
                        pred = sample_logits(logits)
                        preds.append(pred)
                    
                    x_t = torch.stack(preds, dim=-1)
                    last_tokens = x_t  # Update for next iteration
            
            all_outputs = [torch.stack(out, dim=1) for out in all_outputs]
            return all_outputs
        
        else:
            # Default: zeros (will produce garbage)
            all_outputs = []
            for i in range(self.chunks):
                chunk_hidden = high_hidden[:, i, :].unsqueeze(0).repeat(self.low_rnn.num_layers, 1, 1)
                x_input = torch.zeros(batch_size, chunk_len, 8, dtype=torch.long, device=self.device)
                embeds = [self.embeddings[j](x_input[:, :, j]) for j in range(8)]
                x_emb = torch.cat(embeds, dim=-1)
                high_context = chunk_hidden[-1].unsqueeze(1).repeat(1, x_emb.size(1), 1)
                low_input = torch.cat([x_emb, high_context], dim=-1)
                h_dec, _ = self.low_rnn(low_input, chunk_hidden)
                h_dec = self.dropout(h_dec)  # Apply dropout
                outputs = [layer(h_dec) for layer in self.output_layers]
                
                if i == 0:
                    all_outputs = outputs
                else:
                    for j in range(8):
                        all_outputs[j] = torch.cat([all_outputs[j], outputs[j]], dim=1)
            return all_outputs

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