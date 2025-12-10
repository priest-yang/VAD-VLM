import torch
import torch.nn as nn
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 commitment_cost: float = 0.25,
                 **kwargs):
        super(VectorQuantizer, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1. / num_embeddings, 1. / num_embeddings)

    def forward(self, x):
        # Flatten input
        flat_x = x.reshape(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_x**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_x, self.embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and calculate loss
        quantized = torch.matmul(encodings, self.embedding.weight).view(x.shape)
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity


class VQVAEEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 embedding_dim,
                 **kwargs):
        super(VQVAEEncoder, self).__init__(**kwargs)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class VQVAEDecoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 output_dim,
                 **kwargs):
        super(VQVAEDecoder, self).__init__(**kwargs)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class VQVAE(nn.Module):
    """Implements the VQVAE module. 
    Args: 
    """

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 embedding_dim,
                 num_embeddings,
                 commitment_cost,
                 **kwargs):
        super(VQVAE, self).__init__(**kwargs)
        self.encoder = VQVAEEncoder(input_dim, hidden_dim, embedding_dim)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = VQVAEDecoder(embedding_dim, hidden_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, perplexity = self.quantizer(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, perplexity, z
