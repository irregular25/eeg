''' Implementation of the SSM model s4, from "Efficiently Modeling Long Sequences with Structured State Spaces", Gu, Goel, Re (2022)
Note that the model is heavy to compute. The following code is greatly inspired by the Annotated s4'''

import torch
from torch.utils.data import DataLoader, Dataset

from functools import partial
import jax
import jax.numpy as np # not for numpy !
from flax import linen as nn
from flax.training import train_state # , checkpoints
from jax.nn.initializers import lecun_normal, normal
from jax.numpy.linalg import eigh, inv, matrix_power
from jax.scipy.signal import convolve
import optax
from tqdm import tqdm

import mne
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.channels import make_standard_montage
from mne.decoding import FilterEstimator
import matplotlib.pyplot as plt
import numpy

# Hyperparameters
n_classes = 2
n_epochs = 10
l_max = 400*64 # (n_samples * n_channels) = 25600: sequence max

d_model = 40 # 64
n_layers = 4 # number of layers
dropout=0.0
prenorm = True
embedding=False
N = 32 #64 # hyperparameter for the S4 layer


## Core S4 Layer ##
class S4Layer(nn.Module):
    N: int = N
    l_max: int = l_max
    decode: bool = False

    # Special parameters with multiplicative factor on lr and no weight decay (handled by main train script)
    lr = {
        "Lambda_re": 0.1,
        "Lambda_im": 0.1,
        "P": 0.1,
        "B": 0.1,
        "log_step": 0.1,
    }

    def setup(self):
        # Learned Parameters (C is complex!)
        init_A_re, init_A_im, init_P, init_B = hippo_initializer(self.N)
        self.Lambda_re = self.param("Lambda_re", init_A_re, (self.N,))
        self.Lambda_im = self.param("Lambda_im", init_A_im, (self.N,))
        # Ensure the real part of Lambda is negative
        # (described in the SaShiMi follow-up to S4)
        self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        self.P = self.param("P", init_P, (self.N,))
        self.B = self.param("B", init_B, (self.N,))
        # C should be init as standard normal
        # This doesn't work due to how JAX handles complex optimizers https://github.com/deepmind/optax/issues/196
        # self.C = self.param("C", normal(stddev=1.0, dtype=np.complex64), (self.N,))
        self.C = self.param("C", normal(stddev=0.5**0.5), (self.N, 2))
        self.C = self.C[..., 0] + 1j * self.C[..., 1]
        self.D = self.param("D", nn.initializers.ones, (1,))
        self.step = np.exp(self.param("log_step", log_step_initializer(), (1,)))

        if not self.decode:
            # CNN mode, compute kernel.
            self.K = kernel_DPLR(
                self.Lambda,
                self.P,
                self.P,
                self.B,
                self.C,
                self.step,
                self.l_max,
            )

        else:
            # RNN mode, discretize

            # Flax trick to cache discrete form during decoding.
            def init_discrete():
                return discrete_DPLR(
                    self.Lambda,
                    self.P,
                    self.P,
                    self.B,
                    self.C,
                    self.step,
                    self.l_max,
                )

            ssm_var = self.variable("prime", "ssm", init_discrete)
            if self.is_mutable_collection("prime"):
                ssm_var.value = init_discrete()
            self.ssm = ssm_var.value

            # RNN Cache
            self.x_k_1 = self.variable(
                "cache", "cache_x_k", np.zeros, (self.N,), np.complex64
            )

    def __call__(self, u):
        # This is identical to SSM Layer
        if not self.decode:
            # CNN Mode
            return causal_convolution(u, self.K) + self.D * u
        else:
            # RNN Mode
            x_k, y_s = scan_SSM(*self.ssm, u[:, np.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u

# Build Layers and Model Architecture

def scan_SSM(Ab, Bb, Cb, u, x0):
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)

def cloneLayer(layer):
    return nn.vmap(
        layer,
        in_axes=1,
        out_axes=1,
        variable_axes={"params": 1, "cache": 1, "prime": 1},
        split_rngs={"params": True},
    )

SSMLayer = cloneLayer(SSMLayer)

class SequenceBlock(nn.Module):
    layer_cls: nn.Module
    #layer: dict  # Hyperparameters of inner layer
    dropout: float
    d_model: int
    prenorm: bool = True
    glu: bool = True
    training: bool = True
    decode: bool = False

    def setup(self):
        #self.seq = self.layer_cls(**self.layer, decode=self.decode)
        self.seq = self.layer_cls(decode=self.decode)
        self.norm = nn.LayerNorm()
        self.out = nn.Dense(self.d_model)
        if self.glu:
            self.out2 = nn.Dense(self.d_model)
        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def __call__(self, x):
        skip = x
        if self.prenorm:
            x = self.norm(x)
        x = self.seq(x)
        x = self.drop(nn.gelu(x))
        if self.glu:
            x = self.out(x) * jax.nn.sigmoid(self.out2(x))
        else:
            x = self.out(x)
        x = skip + self.drop(x)
        if not self.prenorm:
            x = self.norm(x)
        return x

class Embedding(nn.Embed):
    num_embeddings: int
    features: int

    @nn.compact
    def __call__(self, x):
        y = nn.Embed(self.num_embeddings, self.features)(x[..., 0])
        return np.where(x > 0, y, 0.0)

class StackedModel(nn.Module):
    layer_cls: nn.Module
    #layer: dict  # Extra arguments to pass into layer constructor
    d_output: int
    d_model: int
    n_layers: int
    prenorm: bool = True
    dropout: float = 0.0
    embedding: bool = False  # Use nn.Embed instead of nn.Dense encoder
    classification: bool = True # False
    training: bool = True
    decode: bool = False  # Probably should be moved into layer_args

    def setup(self):
        if self.embedding:
            self.encoder = Embedding(self.d_output, self.d_model)
        else:
            self.encoder = nn.Dense(self.d_model)
        self.decoder = nn.Dense(self.d_output)
        self.layers = [
            SequenceBlock(
                layer_cls=self.layer_cls,
                #layer=self.layer,
                prenorm=self.prenorm,
                d_model=self.d_model,
                dropout=self.dropout,
                training=self.training,
                decode=self.decode,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x):
        if not self.classification:
            if not self.embedding:
                x = x / 255.0  # Normalize
            if not self.decode:
                x = np.pad(x[:-1], [(1, 0), (0, 0)])
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        if self.classification:
            x = np.mean(x, axis=0)
        x = self.decoder(x)
        return nn.log_softmax(x, axis=-1)

BatchStackedModel = nn.vmap(
    StackedModel,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "dropout": None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True},
)

## Compute HIPPO matrix ##
# addressing issues like vanishing/exploding gradients problem for long sequence
# Essential for good performance with s4

def log_step_initializer(dt_min=0.001, dt_max=0.1):
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init

def init(x):
    def _init(key, shape):
        assert shape == x.shape
        return x

    return _init

def hippo_initializer(N):
    Lambda, P, B, _ = make_DPLR_HiPPO(N)
    return init(Lambda.real), init(Lambda.imag), init(P), init(B)

def make_HiPPO(N):
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A

def make_NPLR_HiPPO(N):
    # Make HiPPO
    nhippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return nhippo, P, B

def make_DPLR_HiPPO(N):
    """Diagonalize NPLR representation"""
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, np.newaxis] * P[np.newaxis, :]

    # Check skew symmetry
    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)
    # assert np.allclose(Lambda_real, S_diag, atol=1e-3)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = eigh(S * -1j)

    P = V.conj().T @ P
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V

## Diagonal Plus Low Rank and Normal Plus Low Rank Matrices ##
@jax.jit
def cauchy(v, omega, lambd):
    """Cauchy matrix multiplication: (n), (l), (n) -> (l)"""
    cauchy_dot = lambda _omega: (v / (_omega - lambd)).sum()
    return jax.vmap(cauchy_dot)(omega)

def kernel_DPLR(Lambda, P, Q, B, C, step, L):
    # Evaluate at roots of unity
    # Generating function is (-)z-transform, so we evaluate at (-)root
    Omega_L = np.exp((-2j * np.pi) * (np.arange(L) / L))

    aterm = (C.conj(), Q.conj())
    bterm = (B, P)

    g = (2.0 / step) * ((1.0 - Omega_L) / (1.0 + Omega_L))
    c = 2.0 / (1.0 + Omega_L)

    # Reduction to core Cauchy kernel
    k00 = cauchy(aterm[0] * bterm[0], g, Lambda)
    k01 = cauchy(aterm[0] * bterm[1], g, Lambda)
    k10 = cauchy(aterm[1] * bterm[0], g, Lambda)
    k11 = cauchy(aterm[1] * bterm[1], g, Lambda)
    atRoots = c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
    out = np.fft.ifft(atRoots, L).reshape(L)
    return out.real

def discrete_DPLR(Lambda, P, Q, B, C, step, L):
    # Convert parameters to matrices
    B = B[:, np.newaxis]
    Ct = C[np.newaxis, :]

    N = Lambda.shape[0]
    A = np.diag(Lambda) - P[:, np.newaxis] @ Q[:, np.newaxis].conj().T
    I = np.eye(N)

    # Forward Euler
    A0 = (2.0 / step) * I + A

    # Backward Euler
    D = np.diag(1.0 / ((2.0 / step) - Lambda))
    Qc = Q.conj().T.reshape(1, -1)
    P2 = P.reshape(-1, 1)
    A1 = D - (D @ P2 * (1.0 / (1 + (Qc @ D @ P2))) * Qc @ D)

    # A bar and B bar
    Ab = A1 @ A0
    Bb = 2 * A1 @ B

    # Recover Cbar from Ct
    Cb = Ct @ inv(I - matrix_power(Ab, L)).conj()
    return Ab, Bb, Cb.conj()

# compute convolution #
def causal_convolution(u, K, nofft=False):
    if nofft:
        return convolve(u, K, mode="full")[: u.shape[0]]
    else:
        assert K.shape[0] == u.shape[0]
        ud = np.fft.rfft(np.pad(u, (0, K.shape[0])))
        Kd = np.fft.rfft(np.pad(K, (0, u.shape[0])))
        out = ud * Kd
        return np.fft.irfft(out)[: u.shape[0]]


### Functions for Training and validation ###

@partial(np.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label):
    one_hot_label = jax.nn.one_hot(label, num_classes=logits.shape[0])
    return -np.sum(one_hot_label * logits)


@partial(np.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return np.argmax(logits) == label


# As we're using Flax, we also write a utility function to return a default TrainState object.
# This function initializes model parameters, as well as our optimizer. Note that for S4 models,
# we use a custom learning rate for parameters of the S4 kernel (lr = 0.001, no weight decay).
def map_nested_fn(fn):
    """Recursively apply `fn to the key-value pairs of a nested dict / pytree."""

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn

def create_train_state(
    rng,
    model_cls,
    trainloader,
    lr=1e-3,
    lr_layer=None,
    lr_schedule=False,
    weight_decay=0.0,
    total_steps=-1,
):
    model = model_cls(training=True)
    init_rng, dropout_rng = jax.random.split(rng, num=2)
    params = model.init(
        {"params": init_rng, "dropout": dropout_rng},
        np.array(next(iter(trainloader))[0].numpy()),
    )
    
    #params = params["params"].unfreeze() # better with Optax ?
    params = params["params"]
    # Debugging: Use Optax.multi_transform ?

    if lr_schedule:
        schedule_fn = lambda lr: optax.cosine_onecycle_schedule(
            peak_value=lr,
            transition_steps=total_steps,
            pct_start=0.1,
        )
    else:
        schedule_fn = lambda lr: lr
    # lr_layer is a dictionary from parameter name to LR multiplier
    if lr_layer is None:
        lr_layer = {}

    optimizers = {
        k: optax.adam(learning_rate=schedule_fn(v * lr))
        for k, v in lr_layer.items()
    }
    # Add default optimizer
    # Note: it would be better to use a dummy key such as None that can't conflict with parameter names,
    # but this causes a hard-to-trace error; it seems that the transforms keys list is being sorted inside optax.multi_transform
    # which causes an error since None can't be compared to str
    optimizers["__default__"] = optax.adamw(
        learning_rate=schedule_fn(lr),
        weight_decay=weight_decay,
    )
    name_map = map_nested_fn(lambda k, _: k if k in lr_layer else "__default__")
    tx = optax.multi_transform(optimizers, name_map)

    # Check that all special parameter names are actually parameters
    extra_keys = set(lr_layer.keys()) - set(jax.tree_leaves(name_map(params)))
    assert (
        len(extra_keys) == 0
    ), f"Special params {extra_keys} do not correspond to actual params"

    # Print parameter count
    _is_complex = lambda x: x.dtype in [np.complex64, np.complex128]
    param_sizes = map_nested_fn(
        lambda k, param: param.size * (2 if _is_complex(param) else 1)
        if lr_layer.get(k, lr) > 0.0
        else 0
    )(params)
    print(f"[*] Trainable Parameters: {sum(jax.tree_leaves(param_sizes))}")
    print(f"[*] Total training steps: {total_steps}")

    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

## Generic train and validation functions ##
# arguments: training state, model class, dataloader, model-specific step function.

def train_epoch(state, rng, model, trainloader, classification=True):
    # Store Metrics
    model = model(training=True)
    batch_losses, batch_accuracies = [], []
    for batch_idx, (inputs, labels) in enumerate(tqdm(trainloader)):
        inputs = np.array(inputs.numpy())
        labels = np.array(labels.numpy())  # Not the most efficient...
        rng, drop_rng = jax.random.split(rng)
        state, loss, acc = train_step(state,drop_rng,inputs,labels,model,classification=classification)
        batch_losses.append(loss)
        batch_accuracies.append(acc)

    # Return average loss over batches
    return (
        state,
        np.mean(np.array(batch_losses)),
        np.mean(np.array(batch_accuracies)),
    )

@partial(jax.jit, static_argnums=(4, 5))
def train_step(state, rng, batch_inputs, batch_labels, model, classification=True):
    def loss_fn(params):
        logits, mod_vars = model.apply(
            {"params": params},
            batch_inputs,
            rngs={"dropout": rng},
            mutable=["intermediates"],
        )
        loss = np.mean(cross_entropy_loss(logits, batch_labels))
        acc = np.mean(compute_accuracy(logits, batch_labels))
        return loss, (logits, acc)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, acc)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, acc


@partial(jax.jit, static_argnums=(3, 4))
def eval_step(batch_inputs, batch_labels, params, model, classification=True):
    logits = model.apply({"params": params}, batch_inputs)
    loss = np.mean(cross_entropy_loss(logits, batch_labels))
    acc = np.mean(compute_accuracy(logits, batch_labels))
    return loss, acc

def validate(params, model, testloader, classification=True):
    # Compute average loss & accuracy
    model = model(training=False)
    losses, accuracies = [], []
    for batch_idx, (inputs, labels) in enumerate(tqdm(testloader)):
        inputs = np.array(inputs.numpy())
        labels = np.array(labels.numpy())  # Not the most efficient...
        loss, acc = eval_step(inputs, labels, params, model, classification=classification)
        losses.append(loss)
        accuracies.append(acc)
    return np.mean(np.array(losses)), np.mean(np.array(accuracies))


### MAIN LOOP###
def S4train(X, y, X_test, y_test):
    seed = 0
    torch.random.manual_seed(seed)  # For dataloader order
    key = jax.random.PRNGKey(seed)
    key, rng, train_rng = jax.random.split(key, num=3)

    # Check if classification dataset
    classification = True

    ''' Create dataset '''
    Xt = torch.FloatTensor(X)
    labels = torch.FloatTensor(y)
    train_data = Xt.view(Xt.size(0), l_max, 1) # expected shape : (batch_size, sequance_length, 1)
    print(train_data.size())
    dataset = EEGDataset(train_data, labels)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    te = torch.FloatTensor(X_test)
    test_data = te.view(te.size(0), l_max, 1)
    labs = torch.FloatTensor(y_test)
    datatest = EEGDataset(test_data, labs)
    test_loader = DataLoader(datatest, batch_size=64, shuffle=False)

    # Get model class and arguments
    layer_cls = S4Layer

    # Extract custom hyperparameters from model class
    lr_layer = getattr(layer_cls, "lr", None)

    print("Training")

    model_cls = partial(BatchStackedModel, layer_cls=layer_cls, d_output=n_classes, classification=classification, d_model = d_model,
        n_layers = n_layers,
    )

    state = create_train_state(rng, model_cls, train_loader, lr=0.001, lr_layer=lr_layer, lr_schedule=False, weight_decay=0.01,
        total_steps=len(train_loader) * n_epochs,
    )

    # Loop over epochs
    best_loss, best_acc, best_epoch = 10000, 0, 0
    for epoch in range(n_epochs):
        print(f"[*] Starting Training Epoch {epoch + 1}...")
        state, train_loss, train_acc = train_epoch(
            state,
            train_rng,
            model_cls,
            train_loader,
            classification=classification,
        )

        print(f"[*] Running Epoch {epoch + 1} Validation...")
        test_loss, test_acc = validate(state.params, model_cls, test_loader, classification=classification)

        print(f"\n=>> Epoch {epoch + 1} Metrics ===")
        print(
            f"\tTrain Loss: {train_loss:.5f} -- Train Accuracy:"
            f" {train_acc:.4f}\n\t Test Loss: {test_loss:.5f} --  Test"
            f" Accuracy: {test_acc:.4f}"
        )

### Load Dataset ###
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.labels = y
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        return self.data[id], self.labels[id]

def retrieve_data(subject_start,subject_end):
    # retrieve X,y for train or test dataset
    subject_list = [i for i in range(subject_start + 1,subject_end)]
    raw_fnames10 = eegbci.load_data(subject_start, runs)
    all_raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames10])
    for subject in subject_list:
        raw_fnames = eegbci.load_data(subject, runs)
        raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
        all_raw = concatenate_raws([all_raw,raw])

    # standardize
    eegbci.standardize(all_raw)  
    # standard montage
    montage = make_standard_montage("standard_1005")
    all_raw.set_montage(montage)
    
    # Retrieve Events
    events, event_id = mne.events_from_annotations(all_raw)

    tmin, tmax = -0.245, 2.25  # smaller epochs to reduce computation time
    #tmin, tmax = -1, 4
    picks = mne.pick_types(all_raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    epochs = mne.Epochs(all_raw, events, event_id=[2,3], tmin=tmin, tmax=tmax,baseline=None,preload=True,picks=picks)
    #,reject=dict(eeg=150e-6)) # rejection threshold too high
    print(epochs)

    # EEG signals: n_epochs, n_eeg_channels, n_times
    epochs_data = epochs.get_data(copy=False)  
    # labels 0 or 1: hands vs feet
    labels = epochs.events[:, -1] - 2

    ### Preprocessing step ###
    # Apply band-pass filter
    filt = FilterEstimator(epochs.info, 7.0, 30, fir_design='firwin')           
    X = epochs_data
    X = filt.transform(X)
    y = labels
    return X,y


if __name__ == '__main__':
    # Choose and load Datasets
    runs=[6,10,14]
    X,y = retrieve_data(3,8)
    X_val,y_val = retrieve_data(1,2)
    # train model
    S4train(X,y,X_val,y_val)
