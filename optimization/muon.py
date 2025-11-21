"""
Muon Optimizer (Momentum Updated Orthogonal Newton).
"""

import torch
import torch.distributed as dist

def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power of the matrix G
    (i.e. the orthogonal matrix U such that G = U*S*V^T -> UV^T).
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - Momentum Updated Orthogonal Newton
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.ndim < 2:
                    # Muon only works for 2D+ tensors. 
                    # For 1D tensors (biases, norms), use standard SGD/Adam behavior or skip?
                    # Usually Muon is paired with AdamW for 1D params.
                    # Here we assume the user separates params.
                    # If we encounter < 2D, we'll just do simple SGD.
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(p)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(grad)
                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf
                    p.data.add_(grad, alpha=-lr)
                    continue

                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)

                if nesterov:
                    g = grad.add(buf, alpha=momentum)
                else:
                    g = buf

                # Orthogonalization
                if g.ndim > 2:
                    # Flatten to 2D
                    g_flat = g.view(g.size(0), -1)
                else:
                    g_flat = g

                update = zeropower_via_newtonschulz5(g_flat, steps=ns_steps)

                if g.ndim > 2:
                    update = update.view_as(g)

                update *= max(1, g_flat.size(0)/g_flat.size(1))**0.5
                
                p.data.add_(update, alpha=-lr)

        return loss
