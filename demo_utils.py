from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.linalg import toeplitz


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "demo_results"


def load_mat_array(path, variable):
    """Load one named array from a MATLAB .mat file."""
    data = sio.loadmat(path)
    if variable not in data:
        public_keys = [key for key in data if not key.startswith("__")]
        raise KeyError(f"{variable!r} not found in {path}. Available variables: {public_keys}")
    return np.asarray(data[variable])


def to_interval_trace_time(cube):
    """Convert MATLAB layout [time, trace, interval] to [interval, trace, time]."""
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3-D cube, got shape {cube.shape}")
    return np.transpose(cube, (2, 1, 0))


def normalize_traces(data, eps=1e-8):
    """Normalize an array with a global mean/std and protect constant inputs."""
    data = np.asarray(data, dtype=np.float32)
    mean = np.mean(data, keepdims=True)
    std = np.std(data, keepdims=True)
    std = np.maximum(std, eps)
    return (data - mean) / std, mean, std


def extract_positions(data, threshold=1e-8):
    """Return a binary reflection-position mask from amplitudes."""
    return (np.abs(np.asarray(data)) > threshold).astype(np.int64)


def pick_section(cube, interval=9, trace_start=0, trace_count=30, time_start=200, time_stop=800):
    """Pick a [time, trace] section from a [interval, trace, time] cube."""
    trace_stop = trace_start + trace_count
    return np.asarray(cube[interval, trace_start:trace_stop, time_start:time_stop]).T


def pick_interval_section(cube, trace=100, interval_start=0, interval_count=30, time_start=200, time_stop=800):
    """Pick a [time, interval] section at one trace from a [interval, trace, time] cube."""
    interval_stop = interval_start + interval_count
    return np.asarray(cube[interval_start:interval_stop, trace, time_start:time_stop]).T


def wigb(
    section,
    dt=0.001,
    t0=0.0,
    scale=0.75,
    ax=None,
    color="black",
    linewidth=0.45,
    fill=True,
    panel_label=None,
):
    """Plot a seismic section in simple wiggle-trace style.

    Parameters
    ----------
    section:
        2-D array in [time, trace] order.
    dt:
        Sampling interval in seconds.
    t0:
        Start time in seconds.
    """
    import matplotlib.pyplot as plt

    section = np.asarray(section, dtype=np.float32)
    if section.ndim != 2:
        raise ValueError(f"Expected [time, trace] section, got shape {section.shape}")
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4.2))

    nt, ntraces = section.shape
    time_ms = (t0 + np.arange(nt) * dt) * 1000.0
    max_amp = np.nanmax(np.abs(section), axis=0, keepdims=True)
    max_amp[max_amp == 0] = 1.0
    normalized = section / max_amp * scale

    for i in range(ntraces):
        center = i + 1
        trace = normalized[:, i]
        ax.plot(center + trace, time_ms, color=color, linewidth=linewidth)
        if fill:
            ax.fill_betweenx(time_ms, center, center + trace, where=trace > 0, color=color, alpha=0.35)

    ax.set_xlim(0.5, ntraces + 0.5)
    ax.set_ylim(time_ms[-1], time_ms[0])
    ax.set_xlabel("Trace no.")
    ax.set_ylabel("Time (ms)")
    ax.tick_params(direction="in", top=True, right=True, labelsize=8)
    if panel_label:
        ax.text(-0.12, 1.02, panel_label, transform=ax.transAxes, fontsize=10)
    return ax


def plot_position_points(
    section,
    ax=None,
    color="black",
    size=4,
    panel_label=None,
    dt=0.001,
    t0=0.0,
    label="Time (ms)",
):
    """Plot binary reflection positions for a [time, trace] section."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4.2))
    mask = extract_positions(section)
    time_idx, trace_idx = np.nonzero(mask)
    y = (t0 + time_idx * dt) * 1000.0
    y_max = (t0 + section.shape[0] * dt) * 1000.0
    y_min = t0 * 1000.0
    ax.scatter(trace_idx + 1, y, c=color, s=size, marker=".", linewidths=0)
    ax.set_xlim(0.5, section.shape[1] + 0.5)
    ax.set_ylim(y_max, y_min)
    ax.set_xlabel("Trace no.")
    ax.set_ylabel(label)
    ax.tick_params(direction="in", top=True, right=True, labelsize=8)
    if panel_label:
        ax.text(-0.12, 1.02, panel_label, transform=ax.transAxes, fontsize=10)
    return ax


def save_mat(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(path, arrays)
    return path


def ricker_wavelet(fm=30.0, dt=0.001, length=0.08):
    """Return a zero-phase Ricker wavelet."""
    half = length / 2.0
    t = np.arange(-half, half + dt, dt, dtype=np.float32)
    arg = np.pi * fm * t
    wavelet = (1.0 - 2.0 * arg**2) * np.exp(-(arg**2))
    return wavelet.astype(np.float32)


def wavelet_matrix(nsamples, fm=30.0, dt=0.001, length=0.08):
    """Build a convolution matrix matching the MATLAB demo layout."""
    wavelet = ricker_wavelet(fm=fm, dt=dt, length=length).reshape(-1, 1)
    padded = np.vstack([wavelet, np.zeros((nsamples - 1, 1), dtype=np.float32)])
    matrix = toeplitz(padded[:, 0], np.zeros(nsamples, dtype=np.float32))
    half = (len(wavelet) - 1) // 2
    return matrix[half : half + nsamples].astype(np.float32)


def soft_threshold(values, threshold):
    """Soft-threshold operator used by the sparse deconvolution demo."""
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def sparse_spike_deconvolution(section, fm=30.0, dt=0.001, lam=0.002, n_iter=80):
    """Simple ISTA sparse-spike deconvolution for a [time, trace] seismic section."""
    section = np.asarray(section, dtype=np.float32)
    nsamples, _ = section.shape
    wave = wavelet_matrix(nsamples, fm=fm, dt=dt)
    lipschitz = np.linalg.norm(wave, ord=2) ** 2 + 1e-6
    step = 1.0 / lipschitz
    result = np.zeros_like(section)
    wt = wave.T
    for _ in range(int(n_iter)):
        gradient = wt @ (wave @ result - section)
        result = soft_threshold(result - step * gradient, lam * step)
    return result


def train_reflectivity_demo(
    epochs=10,
    interval=9,
    train_traces=2,
    data_name="seis",
    label_name="rbp_180_200",
    learning_rate=1e-3,
    device=None,
):
    """Run a small MscaleCNN training demo and return full-trace prediction."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    from core.Mscale_cnnnet import MscaleCNN

    torch.manual_seed(10)
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    seis = to_interval_trace_time(load_mat_array(DATA_DIR / f"{data_name}.mat", data_name))
    labels = to_interval_trace_time(load_mat_array(DATA_DIR / f"{label_name}.mat", label_name))
    x_all = seis[interval].astype(np.float32)
    y_all = labels[interval].astype(np.float32)
    x_norm, x_mean, x_std = normalize_traces(x_all)
    y_norm, y_mean, y_std = normalize_traces(y_all)

    x_train = torch.from_numpy(x_norm[:train_traces])
    y_train = torch.from_numpy(y_norm[:train_traces])
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=1, shuffle=False)

    model = MscaleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()
    losses = []

    for _ in range(int(epochs)):
        epoch_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred, dec1, dec2 = model(x_batch)
            target2 = F.interpolate(y_batch.view(-1, 1, y_batch.shape[-1]), scale_factor=2, mode="linear")
            loss = criterion(pred, y_batch) + 0.1 * criterion(dec1, y_batch)
            loss = loss + 0.1 * criterion(dec2, target2.view(dec2.shape))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
        losses.append(epoch_loss / len(loader))

    predictions = []
    model.eval()
    with torch.no_grad():
        for trace in torch.from_numpy(x_norm).to(device):
            pred, _, _ = model(trace.unsqueeze(0))
            predictions.append(pred.squeeze(0).cpu().numpy())
    pred_norm = np.stack(predictions, axis=0)
    pred = pred_norm * y_std + y_mean

    out_path = save_mat(
        RESULT_DIR / "quick_mscale_reflectivity.mat",
        prediction=pred.astype(np.float32),
        loss=np.asarray(losses, dtype=np.float32),
        interval=np.asarray([[interval + 1]]),
    )
    return {"prediction": pred, "loss": np.asarray(losses), "model": model, "path": out_path}


def train_position_demo(
    epochs=10,
    interval=9,
    train_traces=200,
    data_name="seis",
    learning_rate=1e-3,
    device=None,
):
    """Run a small DCNN_C position-classification demo and return a binary mask."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from core.Mscale_cnnnet import DCNN_C

    torch.manual_seed(10)
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    seis = to_interval_trace_time(load_mat_array(DATA_DIR / f"{data_name}.mat", data_name))
    ref = to_interval_trace_time(load_mat_array(DATA_DIR / "ref.mat", "ref"))
    x_all = seis[interval].astype(np.float32)
    y_all = extract_positions(ref[interval], threshold=1e-20).astype(np.int64)
    x_norm, _, _ = normalize_traces(x_all)

    x_train = torch.from_numpy(x_norm[:train_traces])
    y_train = torch.from_numpy(y_all[:train_traces])
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=min(train_traces, 200), shuffle=True)

    model = DCNN_C().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    losses = []

    for _ in range(int(epochs)):
        epoch_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)[0]
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
        losses.append(epoch_loss / len(loader))

    predictions = []
    model.eval()
    with torch.no_grad():
        for trace in torch.from_numpy(x_norm).to(device):
            logits = model(trace.unsqueeze(0))[0]
            predictions.append(torch.argmax(logits, dim=1).squeeze(0).cpu().numpy())
    mask = np.stack(predictions, axis=0).astype(np.int64)

    out_path = save_mat(
        RESULT_DIR / "quick_position_mask.mat",
        prediction=mask.astype(np.float32),
        loss=np.asarray(losses, dtype=np.float32),
        interval=np.asarray([[interval + 1]]),
    )
    return {"prediction": mask, "loss": np.asarray(losses), "model": model, "path": out_path}
