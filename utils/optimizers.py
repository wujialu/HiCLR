""" Optimizers class """
import torch
import torch.optim as argsim
from torch.nn.utils import clip_grad_norm_

def build_optim(model, args):
    """ Build optimizer """
    argsim = Optimizer(
        args.optim, args.learning_rate, args.max_grad_norm,
        lr_decay=args.learning_rate_decay,
        start_decay_steps=args.start_decay_steps,
        decay_steps=args.decay_steps,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        adagrad_accum=args.adagrad_accumulator_init,
        decay_method=args.decay_method,
        warmup_steps=args.warmup_steps,
        model_size=args.d_model)

    argsim.set_parameters(model.named_parameters())

    return argsim


class MultipleOptimizer(object):
    """ Implement multiple optimizers needed for sparse adam """

    def __init__(self, op):
        """ ? """
        self.optimizers = op

    def zero_grad(self):
        """ ? """
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        """ ? """
        for op in self.optimizers:
            op.step()

    @property
    def state(self):
        """ ? """
        return {k: v for op in self.optimizers for k, v in op.state.items()}

    def state_dict(self):
        """ ? """
        return [op.state_dict() for op in self.optimizers]

    def load_state_dict(self, state_dicts):
        """ ? """
        assert len(state_dicts) == len(self.optimizers)
        for i in range(len(state_dicts)):
            self.optimizers[i].load_state_dict(state_dicts[i])


class Optimizer(object):
    """
    Controller class for argsimization. Mostly a thin
    wrapper for `argsim`, but also useful for implementing
    rate scheduling beyond what is currently available.
    Also implements necessary methods for training RNNs such
    as grad manipulations.
    Args:
      method (:obj:`str`): one of [sgd, adagrad, adadelta, adam]
      lr (float): learning rate
      lr_decay (float, optional): learning rate decay multiplier
      start_decay_steps (int, optional): step to start learning rate decay
      beta1, beta2 (float, optional): parameters for adam
      adagrad_accum (float, optional): initialization parameter for adagrad
      decay_method (str, argsion): custom decay argsions
      warmup_steps (int, argsion): parameter for `noam` decay
      model_size (int, argsion): parameter for `noam` decay
    We use the default parameters for Adam that are suggested by
    the original paper https://arxiv.org/pdf/1412.6980.pdf
    These values are also used by other established implementations,
    e.g. https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
    https://keras.io/optimizers/
    Recently there are slightly different values used in the paper
    "Attention is all you need"
    https://arxiv.org/pdf/1706.03762.pdf, particularly the value beta2=0.98
    was used there however, beta2=0.999 is still arguably the more
    established value, so we use that here as well
    """

    def __init__(self, method, learning_rate, max_grad_norm,
                 lr_decay=1, start_decay_steps=None, decay_steps=None,
                 beta1=0.9, beta2=0.999,
                 adagrad_accum=0.0,
                 decay_method=None,
                 warmup_steps=4000,
                 model_size=None):
        self.last_ppl = None
        self.learning_rate = learning_rate
        self.original_lr = learning_rate
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_steps = start_decay_steps
        self.decay_steps = decay_steps
        self.start_decay = False
        self._step = 0
        self.betas = [beta1, beta2]
        self.adagrad_accum = adagrad_accum
        self.decay_method = decay_method
        self.warmup_steps = warmup_steps
        self.model_size = model_size

    def set_parameters(self, params):
        """ ? """
        self.params = []
        self.sparse_params = []
        for k, p in params:
            if p.requires_grad:
                if self.method != 'sparseadam' or "embed" not in k:
                    self.params.append(p)
                else:
                    self.sparse_params.append(p)
        if self.method == 'sgd':
            self.optimizer = argsim.SGD(self.params, lr=self.learning_rate)
        elif self.method == 'adagrad':
            self.optimizer = argsim.Adagrad(self.params, lr=self.learning_rate)
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    self.optimizer.state[p]['sum'] = self.optimizer\
                        .state[p]['sum'].fill_(self.adagrad_accum)
        elif self.method == 'adadelta':
            self.optimizer = argsim.Adadelta(self.params, lr=self.learning_rate)
        elif self.method == 'adam':
            self.optimizer = argsim.Adam(self.params, lr=self.learning_rate,
                                        betas=self.betas, eps=1e-9)
        elif self.method == 'sparseadam':
            self.optimizer = MultipleOptimizer(
                [argsim.Adam(self.params, lr=self.learning_rate,
                            betas=self.betas, eps=1e-8),
                 argsim.SparseAdam(self.sparse_params, lr=self.learning_rate,
                                  betas=self.betas, eps=1e-8)])
        else:
            raise RuntimeError("Invalid argsim method: " + self.method)

    def _set_rate(self, learning_rate):
        self.learning_rate = learning_rate
        if self.method != 'sparseadam':
            self.optimizer.param_groups[0]['lr'] = self.learning_rate
        else:
            for op in self.optimizer.optimizers:
                op.param_groups[0]['lr'] = self.learning_rate

    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def step(self):
        """Update the model parameters based on current gradients.
        Optionally, will employ gradient modification or update learning
        rate.
        """
        self._step += 1

        # Decay method used in tensor2tensor.
        if self.decay_method == "noam":
            self._set_rate(
                self.original_lr *
                (self.model_size ** (-0.5) *
                 min(self._step ** (-0.5),
                     self._step * self.warmup_steps**(-1.5))))
        # Decay based on start_decay_steps every decay_steps
        else:
            if ((self.start_decay_steps is not None) and (
                     self._step >= self.start_decay_steps)):
                self.start_decay = True
            if self.start_decay:
                if ((self._step - self.start_decay_steps)
                   % self.decay_steps == 0):
                    self.learning_rate = self.learning_rate * self.lr_decay

        if self.method != 'sparseadam':
            self.optimizer.param_groups[0]['lr'] = self.learning_rate

        if self.max_grad_norm:
            clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()
        
    def state_dict(self):
        """ ? """
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """ ? """
        self.optimizer.load_state_dict(state_dict)