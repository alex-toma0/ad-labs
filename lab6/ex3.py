import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import scipy.io
import numpy as np
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import from_scipy_sparse_matrix

class Encoder(nn.Module):
    def __init__(self, input_dim):
        super(Encoder, self).__init__()
        self.gc1 = GCNConv(input_dim, 128)
        self.gc2 = GCNConv(128, 64)
    
    def forward(self, x, edge_index):
        x = F.relu(self.gc1(x, edge_index))
        x = F.relu(self.gc2(x, edge_index))
        return x
    
class AttributeDecoder(nn.Module):
    def __init__(self, input_dim):
        super(AttributeDecoder, self).__init__()
        self.fc1 = GCNConv(64, 128)
        self.fc2 = GCNConv(128, input_dim)  
    
    def forward(self, x, edge_index):
        x = F.relu(self.fc1(x, edge_index))
        x = self.fc2(x, edge_index)
        return x
    
class StructureDecoder(nn.Module):
    def __init__(self):
        super(StructureDecoder, self).__init__()
        self.gc = GCNConv(64, 64)
    
    def forward(self, x, edge_index):
        x = F.relu(self.gc(x, edge_index))
        adj = torch.mm(x, x.t())
        return adj

class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(GraphAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim)
        self.attr_decoder = AttributeDecoder(input_dim)
        self.struct_decoder = StructureDecoder()
    
    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        
        x_rec = self.attr_decoder(z, edge_index)
        adj_rec = self.struct_decoder(z, edge_index)
        
        return x_rec, adj_rec
    
def custom_loss(x_original, x_reconstructed, adj_original, adj_reconstructed, alpha=0.8):
    attribute_loss = torch.norm(x_original - x_reconstructed, p='fro')**2
    structure_loss = torch.norm(adj_original - adj_reconstructed, p='fro')**2
    
    return alpha * attribute_loss + (1 - alpha) * structure_loss

if __name__ == '__main__':
    data = scipy.io.loadmat('ACM.mat')
    attributes = torch.tensor(data['Attributes'].toarray(), dtype=torch.float32)
    labels = torch.tensor(data['Label'], dtype=torch.float32)
    adj = data['Network']
    edge_index, _ = from_scipy_sparse_matrix(adj)
    edge_index = edge_index.to(torch.int64)

    model = GraphAutoencoder(input_dim=attributes.shape[1])
    optimizer = torch.optim.Adam(model.parameters(),lr=0.004)

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        x_rec, adj_rec = model(attributes, edge_index)
        loss = custom_loss(attributes, x_rec, 
                           torch.tensor(adj.toarray(), dtype=torch.float32),
                            adj_rec)
        attr_scores = torch.norm(attributes - x_rec, dim=1)
        struct_scores = torch.norm(torch.tensor(adj.toarray(), dtype=torch.float32) - adj_rec, dim=1)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            combined_scores = 0.8 * attr_scores + 0.2 * struct_scores
            
            binary_labels = (labels != 0).float()   
            roc_auc = roc_auc_score(binary_labels.numpy(), combined_scores.detach().numpy())
            print(f'Epoch {epoch+1}, loss: {loss.item():.5f}, ROC AUC Score: {roc_auc:.5f}')

