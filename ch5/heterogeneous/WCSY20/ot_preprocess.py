import numpy as np
import torch
import ot

def compute_barycenter(n, dists, reg):
    A = np.vstack(dists).T
    M = ot.utils.dist0(n)
    M /= M.max()
    weights = np.ones(len(dists)) / len(dists)
    return ot.bregman.barycenter(A, M, reg, weights, numItermax=10000)

def _channelwise_barycenter(arrs, reg=1e-1):
    if arrs.ndim != 4:
        raise ValueError
    if arrs.shape[1] == 3:
        arrs_c = arrs
    elif arrs.shape[-1] == 3:
        arrs_c = arrs.transpose(0, 3, 1, 2)
    else:
        raise ValueError
    _, C, H, W = arrs_c.shape
    n_bins = H * W
    out = []
    for c in range(C):
        dists = []
        for im in arrs_c[:, c]:
            flat = im.reshape(-1)
            s = flat.sum()
            if s == 0:
                flat = np.ones_like(flat) / flat.size
            else:
                flat = flat / s
            dists.append(flat)
        bc = compute_barycenter(n_bins, dists, reg).reshape(H, W)
        out.append(bc)
    return np.stack(out, 0).astype(np.float32)

def local_barycenter(xs, reg=1e-1):
    xs_norm = xs.astype(np.float32) / 255.0
    return _channelwise_barycenter(xs_norm, reg=reg)

def global_barycenter(local_bcs, reg=1e-1):
    stack = np.stack(local_bcs, 0)
    return _channelwise_barycenter(stack, reg=reg)

def _prepare_pixels(x):
    if x.ndimension() != 3 or x.size(0) != 3:
        raise ValueError
    _, H, W = x.shape
    return x.flatten(1).T.cpu().numpy(), H, W
import time
class OTProjector(torch.nn.Module):
    def __init__(self, global_bc, n_samples=512, reg_e=1e-1, random_state=None):
        super().__init__()
        if global_bc.shape[0] != 3:
            raise ValueError
        self.register_buffer('_global_wb', torch.tensor(global_bc.reshape(3, -1).T, dtype=torch.float32))
        self.n_samples = n_samples
        self.reg_e = reg_e
        self.rng = np.random.default_rng(random_state)
    def _sinkhorn_transport(self, xs):
        idx = self.rng.integers(xs.shape[0], size=self.n_samples)
        ot_sink = ot.da.SinkhornTransport(reg_e=self.reg_e, max_iter=10000)
        ot_sink.fit(Xs=xs[idx], Xt=self._global_wb.cpu().numpy())
        return ot_sink.transform(xs)
    def forward(self, x):
        print('running projection...')
        start = time.perf_counter()
        with torch.no_grad():
            orig_shape = x.shape
            x_flat = x.view(-1, *orig_shape[-3:])
            outs = []
            for img in x_flat:
                xs_np, H, W = _prepare_pixels(img)
                transported_np = self._sinkhorn_transport(xs_np)
                transported_t = torch.from_numpy(transported_np).reshape(H, W, 3).permute(2, 0, 1)
                outs.append(transported_t)
            out = torch.stack(outs, 0)
            print(f'took {time.perf_counter() - start}')
            return out.reshape(orig_shape)