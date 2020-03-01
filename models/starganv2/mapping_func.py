# From https://github.com/yunjey/stargan-v2-demo/tree/master/core
import torch
import torch.nn as nn

class MappingNetwork(nn.Module):
    """Mapping network: (latent z, domain y) -> (style s)."""
    def __init__(self, latent_dim=64, style_dim=64, num_domains=2):
        super(MappingNetwork, self).__init__()
        self.num_domains = num_domains
        hidden_dim = 512
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU())
        
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared.append(
                    nn.Linear(hidden_dim, style_dim))

    def forward(self, z, y):
        """
        Inputs:
            - z: latent vectors of shape (batch, latent_dim).
            - y: domain labels of shape (batch).
        Output:
            - s: style vectors of shape (batch, style_dim).
        """
        #z = z / torch.norm(z, p=2, dim=1, keepdim=True)
        #z = z / (torch.sqrt(torch.mean(z**2, dim=1, keepdim=True)) + 1e-8)
        h = self.shared(z)
        
        outs = []
        for i in range(self.num_domains):
            out = self.unshared[i](h)        # (batch, style_dim)
            outs.append(out)

        out = torch.stack(outs, dim=1)       # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]                      # (batch, style_dim)
        #print('F_s: ', torch.mean(torch.var(s, dim=0, unbiased=False)))
        return s