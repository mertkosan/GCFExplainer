import torch
import torch_geometric as tg
from tqdm import tqdm
import os

import itertools as it


class EmbedModel(torch.nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dim, output_dim, conv='gin', pool='add', **kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.pre = torch.nn.Linear(self.input_dim, self.hidden_dim)

        if conv == 'gin':
            make_conv = lambda: \
                tg.nn.GINConv(torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_dim, self.hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dim, self.hidden_dim)
                ))
        elif conv == 'gcn':
            make_conv = lambda: \
                tg.nn.GCNConv(self.hidden_dim, self.hidden_dim)
        elif conv == 'sage':
            make_conv = lambda: \
                tg.nn.SAGEConv(self.hidden_dim, self.hidden_dim)
        elif conv == 'gat':
            make_conv = lambda: \
                tg.nn.GATConv(self.hidden_dim, self.hidden_dim)
        else:
            assert False

        self.convs = torch.nn.ModuleList()
        for l in range(self.n_layers):
            self.convs.append(make_conv())

        self.post = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * (self.n_layers + 1), self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.output_dim)
        )

        if pool == 'add':
            self.pool = tg.nn.global_add_pool
        elif pool == 'mean':
            self.pool = tg.nn.global_mean_pool
        elif pool == 'max':
            self.pool = tg.nn.global_max_pool
        elif pool == 'sort':
            self.pool = tg.nn.global_sort_pool
        elif pool == 'att':
            self.pool = tg.nn.GlobalAttention(torch.nn.Sequential(
                torch.nn.Linear(self.hidden_dim * (self.n_layers + 1), self.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_dim, 1)
            ))
        elif pool == 'set':
            self.pool = tg.nn.Set2Set(self.hidden_dim * (self.n_layers + 1), 1)
        self.pool_str = pool

    def forward(self, g):
        x = g.x
        edge_index = g.edge_index

        x = self.pre(x.float())
        emb = x
        xres = x
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            if i & 1:
                x += xres
                xres = x
            x = torch.nn.functional.relu(x)
            emb = torch.cat((emb, x), dim=1)

        x = emb
        if self.pool_str == 'sort':
            x = self.pool(x, g.batch, k=1)
        else:
            x = self.pool(x, g.batch)

        x = self.post(x)
        return x


class SiameseModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.embed_model = None
        self.weighted = False
        self.device = device
        self.n_workers = int(os.cpu_count() / 2)
        self.target_emb = None

    def forward_emb(self, gx, hx):
        raise NotImplementedError

    def forward(self, g, h):
        if self.weighted:
            self.gs = torch.tensor([x.num_nodes for x in g.to_data_list()], device=self.device)
            self.hs = torch.tensor([x.num_nodes for x in h.to_data_list()], device=self.device)
        gx = self.embed_model(g)
        hx = self.embed_model(h)
        return self.forward_emb(gx, hx)

    def predict_inner(self, queries, targets, batch_size=None):
        self = self.to(self.device)
        if batch_size is None or len(queries) <= batch_size:
            # tqdm.write(f'direct predict inner dataset')
            g = tg.data.Batch.from_data_list(queries).to(self.device)
            h = tg.data.Batch.from_data_list(targets).to(self.device)
            with torch.no_grad():
                return self.forward(g, h)
        else:
            # tqdm.write(f'batch predict inner dataset')
            # tqdm.write(f'config.n_workers: {self.n_workers}')
            loader = tg.data.DataLoader(list(zip(queries, targets)), batch_size, num_workers=self.n_workers)
            ret = torch.empty(len(queries), device=self.device)
            for i, (g, h) in enumerate(tqdm(loader, 'batches')):
                g = g.to(self.device)
                h = h.to(self.device)
                with torch.no_grad():
                    ret[i * batch_size:(i + 1) * batch_size] = self.forward(g, h)
            return ret

    def predict_outer(self, queries, targets, batch_size=None):
        self = self.to(self.device)
        if batch_size is None or len(queries) * len(targets) <= batch_size:
            # tqdm.write(f'direct predict outer dataset')
            g = tg.data.Batch.from_data_list(queries).to(self.device)
            h = tg.data.Batch.from_data_list(targets).to(self.device)
            gx = self.embed_model(g)
            hx = self.embed_model(h)
            with torch.no_grad():
                return self.forward_emb(gx[:, None, :], hx)
        else:
            # tqdm.write(f'batch predict outer dataset')
            # tqdm.write(f'config.n_workers: {self.n_workers}')
            g = tg.data.Batch.from_data_list(queries).to(self.device)
            gx = self.embed_model(g)
            loader = tg.data.DataLoader(targets, batch_size // len(queries), num_workers=self.n_workers)
            ret = torch.empty(len(queries), len(targets), device=self.device)
            for i, h in enumerate(tqdm(loader, 'batches')):
                h = h.to(self.device)
                hx = self.embed_model(h)
                with torch.no_grad():
                    ret[:, i * loader.batch_size:(i + 1) * loader.batch_size] = self.forward_emb(gx[:, None, :], hx)
            return ret

    def embed_targets(self, original_graphs):
        self = self.to(self.device)
        self.target_emb = self.embed_model(tg.data.Batch.from_data_list(original_graphs).to(self.device))

    def predict_outer_with_queries(self, queries, batch_size=None):
        with torch.no_grad():
            self = self.to(self.device)
            if batch_size is None or batch_size == len(queries):
                g = tg.data.Batch.from_data_list(queries).to(self.device)
                # g = queries
                gx = self.embed_model(g)
                return self.forward_emb(gx[:, None, :], self.target_emb)
            else:
                loader = tg.data.DataLoader(queries, batch_size=batch_size)
                res_all = []
                for batch in loader:
                    gx_ = self.embed_model(batch.to(self.device))
                    res = self.forward_emb(gx_[:, None, :], self.target_emb)
                    res_all.append(res)
                res_all = torch.cat(res_all, dim=0)
                return res_all

    def criterion(self, lb, ub, pred):
        loss = torch.nn.functional.relu(lb - pred) ** 2 + torch.nn.functional.relu(pred - ub) ** 2
        if self.weighted:
            loss /= ((self.gs + self.hs) / 2) ** 2
        loss = torch.mean(loss)
        return loss


class NormGEDModel(SiameseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(kwargs['device'])
        self.embed_model = EmbedModel(*args, **kwargs)

    def forward_emb(self, gx, hx):
        return torch.norm(gx - hx, dim=-1)
