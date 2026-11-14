
# ================================================================
# FIVE-SEED GPU CONTINUAL LEARNING RESEARCH EXPERIMENT
# Complete CIFAR-100 | PGRFA | Matched replay baseline
# Checkpoint/resume | Automatic PDF and ZIP results
# ================================================================

import os, sys, time, json, math, random, gc, shutil
import zipfile, hashlib, traceback, textwrap
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ======================== CONFIGURATION ========================

SEEDS = [42, 43, 44, 45, 46]
METHODS = ["no_feature", "no_prototype", "no_logit"]

TASKS = 20
CLASSES_PER_TASK = 5
EPOCHS = 18
BATCH_SIZE = 128
REPLAY_BATCH = 96
MEMORY_BUDGET = 2000
MAX_REPLAY_GROUPS = 4

LR = 0.001
MIN_LR = 0.000015
WEIGHT_DECAY = 0.0001
LABEL_SMOOTHING = 0.05

REPLAY_WEIGHT = 0.8
FEATURE_WEIGHT = 0.4
PROTOTYPE_WEIGHT = 0.15
DISTILL_WEIGHT = 0.25
TEMPERATURE = 2.0

# Individual Kaggle executions have time limits.
# Remaining GPU hours do not guarantee a longer single execution.
MAX_SESSION_HOURS = 10.5

WORK = Path("/kaggle/working")
OUT = WORK / "PGRFA_ABLATION_RESEARCH"
DATA = WORK / "cifar100_original"

PDF = WORK / "PGRFA_ABLATION_RESULTS.pdf"
ZIP = WORK / "PGRFA_ABLATION_RESULTS.zip"

OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU not enabled. Kaggle Settings -> Accelerator -> GPU."
    )

DEVICE = torch.device("cuda")
AMP = True

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

MEAN = torch.tensor(
    [0.5071, 0.4867, 0.4408],
    device=DEVICE
).view(1, 3, 1, 1)

STD = torch.tensor(
    [0.2675, 0.2565, 0.2761],
    device=DEVICE
).view(1, 3, 1, 1)

PUBLISHED_ACC = 82.008
PUBLISHED_FORGETTING = 7.115789473684212

START_TIME = time.monotonic()
DEADLINE = START_TIME + MAX_SESSION_HOURS * 3600

print("=" * 70)
print("FIVE-SEED GPU CONTINUAL LEARNING EXPERIMENT")
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("Seeds:", SEEDS)
print("Methods:", METHODS)
print("=" * 70, flush=True)


# ======================== UTILITIES ============================

def log(*args):
    print(
        time.strftime("[%Y-%m-%d %H:%M:%S]"),
        *args,
        flush=True
    )


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".temp")
    temp.write_text(
        json.dumps(obj, indent=2, default=str),
        encoding="utf-8"
    )
    os.replace(temp, path)


def atomic_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".temp")
    df.to_csv(temp, index=False)
    os.replace(temp, path)


def atomic_checkpoint(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".temp")
    torch.save(obj, temp)
    os.replace(temp, path)


CONFIG = {
    "seeds": SEEDS,
    "methods": METHODS,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "replay_batch": REPLAY_BATCH,
    "memory_budget": MEMORY_BUDGET,
    "lr": LR,
    "min_lr": MIN_LR,
    "weight_decay": WEIGHT_DECAY,
    "label_smoothing": LABEL_SMOOTHING,
    "replay_weight": REPLAY_WEIGHT,
    "feature_weight": FEATURE_WEIGHT,
    "prototype_weight": PROTOTYPE_WEIGHT,
    "distill_weight": DISTILL_WEIGHT,
    "temperature": TEMPERATURE,
    "tasks": TASKS,
    "classes_per_task": CLASSES_PER_TASK
}

atomic_json(CONFIG, OUT / "configuration.json")

FINGERPRINT = hashlib.sha256(
    json.dumps(CONFIG, sort_keys=True).encode()
).hexdigest()


# ======================== DATASET ==============================

def prepare_cifar100():

    # First search for an official archive supplied via Add Input.
    archive = DATA / "cifar-100-python.tar.gz"
    extracted = DATA / "cifar-100-python"

    if not (extracted / "train").exists():
        input_root = Path("/kaggle/input")

        if input_root.exists():

            for folder in input_root.rglob("cifar-100-python"):
                if (
                    folder.is_dir()
                    and (folder / "train").exists()
                    and (folder / "test").exists()
                ):
                    shutil.copytree(
                        folder, extracted,
                        dirs_exist_ok=True
                    )
                    break

            if not (extracted / "train").exists():
                for file in input_root.rglob(
                    "cifar-100-python.tar.gz"
                ):
                    shutil.copy2(file, archive)
                    break

    if archive.exists() and not (extracted / "train").exists():
        from torchvision.datasets.utils import extract_archive
        extract_archive(str(archive), str(DATA))

    # Attempt local loading first, then the official download.
    try:
        train = datasets.CIFAR100(
            str(DATA), train=True, download=False
        )
    except Exception:
        log("Downloading official CIFAR-100 dataset")
        train = datasets.CIFAR100(
            str(DATA), train=True, download=True
        )

    test = datasets.CIFAR100(
        str(DATA), train=False, download=False
    )

    assert len(train) == 50000
    assert len(test) == 10000
    assert np.asarray(train.data).shape == (50000, 32, 32, 3)
    assert np.asarray(test.data).shape == (10000, 32, 32, 3)

    train_counts = np.bincount(
        np.asarray(train.targets), minlength=100
    )
    test_counts = np.bincount(
        np.asarray(test.targets), minlength=100
    )

    assert np.all(train_counts == 500)
    assert np.all(test_counts == 100)

    atomic_json({
        "dataset": "Original CIFAR-100",
        "train_images": 50000,
        "test_images": 10000,
        "classes": 100,
        "tasks": 20,
        "classes_per_task": 5,
        "synthetic_images": 0,
        "training_subset": False,
        "evaluation": "Task-incremental, known task ID"
    }, OUT / "dataset_manifest.json")

    log("ORIGINAL CIFAR-100 VERIFIED:",
        len(train), "train,", len(test), "test")

    return train, test


class TaskDataset(Dataset):

    def __init__(self, base, task, augment=False):
        self.base = base
        labels = np.asarray(base.targets)

        self.indices = np.flatnonzero(
            (labels >= task * 5) &
            (labels < (task + 1) * 5)
        )

        self.task = task

        operations = []

        if augment:
            operations += [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip()
            ]

        operations += [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761)
            )
        ]

        self.transform = transforms.Compose(operations)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = int(self.indices[i])
        image = Image.fromarray(self.base.data[index])
        label = int(self.base.targets[index]) - self.task * 5
        return self.transform(image), label, index


def make_loader(base, task, augment=False,
                shuffle=False, seed=0):

    ds = TaskDataset(base, task, augment)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        generator=generator
    )


# ======================== MODEL ================================

class TaskBN(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.BatchNorm2d(channels)
            for _ in range(TASKS)
        ])

        self.active_task = -1

    def forward(self, x, task):
        bn = self.layers[task]

        # Old-task running statistics are frozen.
        training = self.training and task == self.active_task

        return F.batch_norm(
            x,
            bn.running_mean,
            bn.running_var,
            bn.weight,
            bn.bias,
            training=training,
            momentum=bn.momentum,
            eps=bn.eps
        )


class ResidualBlock(nn.Module):

    def __init__(self, input_channels,
                 output_channels, stride=1):

        super().__init__()

        self.conv1 = nn.Conv2d(
            input_channels, output_channels,
            kernel_size=3, stride=stride,
            padding=1, bias=False
        )

        self.bn1 = TaskBN(output_channels)

        self.conv2 = nn.Conv2d(
            output_channels, output_channels,
            kernel_size=3, padding=1, bias=False
        )

        self.bn2 = TaskBN(output_channels)

        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Conv2d(
                input_channels, output_channels,
                kernel_size=1, stride=stride, bias=False
            )
            self.shortcut_bn = TaskBN(output_channels)
        else:
            self.shortcut = None
            self.shortcut_bn = None

    def forward(self, x, task):

        identity = x

        z = F.relu(
            self.bn1(self.conv1(x), task),
            inplace=False
        )

        z = self.bn2(self.conv2(z), task)

        if self.shortcut is not None:
            identity = self.shortcut_bn(
                self.shortcut(identity), task
            )

        return F.relu(z + identity, inplace=False)


class ContinualResNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.stem = nn.Conv2d(
            3, 64, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.stem_bn = TaskBN(64)

        self.blocks = nn.ModuleList([
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 256),
            ResidualBlock(256, 512, 2),
            ResidualBlock(512, 512)
        ])

        self.task_scale = nn.Parameter(
            torch.zeros(TASKS, 512)
        )

        self.task_shift = nn.Parameter(
            torch.zeros(TASKS, 512)
        )

        self.heads = nn.ModuleList([
            nn.Linear(512, 5)
            for _ in range(TASKS)
        ])

    def set_training_task(self, task):
        self.train()
        for module in self.modules():
            if isinstance(module, TaskBN):
                module.active_task = task

    def features(self, x, task):

        z = F.relu(
            self.stem_bn(self.stem(x), task),
            inplace=False
        )

        for block in self.blocks:
            z = block(z, task)

        z = F.adaptive_avg_pool2d(z, 1).flatten(1)

        scale = 1.0 + 0.15 * torch.tanh(
            self.task_scale[task]
        )

        return z * scale + self.task_shift[task]

    def forward(self, x, task, return_features=False):

        features = self.features(x, task)
        logits = self.heads[task](features)

        if return_features:
            return features, logits

        return logits


# ======================== REPLAY MEMORY ========================

class PrototypeMemory:

    def __init__(self):
        self.items = {}
        self.prototypes = {}
        self.cursor = 0

    def __len__(self):
        return sum(len(v) for v in self.items.values())

    def state_dict(self):
        return {
            "items": self.items,
            "prototypes": self.prototypes,
            "cursor": self.cursor
        }

    def load_state_dict(self, state):
        self.items = state["items"]
        self.prototypes = state["prototypes"]
        self.cursor = state.get("cursor", 0)

    @torch.no_grad()
    def update(self, model, train_data, task):

        model.eval()

        loader = make_loader(train_data, task)

        storage = {
            c: {"f": [], "z": [], "i": []}
            for c in range(5)
        }

        for x, y, indices in loader:

            x = x.to(DEVICE)

            features, logits = model(
                x, task, return_features=True
            )

            features = F.normalize(
                features.float(), dim=1
            ).cpu()

            logits = logits.float().cpu()

            for c in range(5):
                mask = (y == c)

                if mask.any():
                    storage[c]["f"].append(features[mask])
                    storage[c]["z"].append(logits[mask])
                    storage[c]["i"].append(indices[mask])

        quota = MEMORY_BUDGET // ((task + 1) * 5)

        # Reduce the allocation for older classes.
        for key in list(self.items.keys()):
            self.items[key] = self.items[key][:quota]

        for c in range(5):

            f = torch.cat(storage[c]["f"])
            z = torch.cat(storage[c]["z"])
            indices = torch.cat(storage[c]["i"])

            assert len(f) == 500

            prototype = F.normalize(
                f.mean(dim=0), dim=0
            )

            distances = 1.0 - f @ prototype

            ordered = distances.argsort()

            # Retain representative samples with feature diversity.
            positions = torch.linspace(
                0, int(0.85 * (len(f) - 1)),
                steps=quota
            ).round().long().unique()

            selected = ordered[positions]

            self.items[(task, c)] = [
                {
                    "index": int(indices[j]),
                    "feature": f[j].half().clone(),
                    "logits": z[j].half().clone()
                }
                for j in selected.tolist()
            ]

            self.prototypes[(task, c)] = (
                prototype.half().clone()
            )

        assert len(self) <= MEMORY_BUDGET

        return len(self)

    def sample(self, batch_size):

        old_tasks = sorted({
            key[0] for key in self.items
            if self.items[key]
        })

        if not old_tasks:
            return []

        n_groups = min(
            MAX_REPLAY_GROUPS, len(old_tasks)
        )

        selected_tasks = [
            old_tasks[
                (self.cursor + j) % len(old_tasks)
            ]
            for j in range(n_groups)
        ]

        self.cursor = (
            self.cursor + n_groups
        ) % len(old_tasks)

        samples = []

        for group_index, task in enumerate(selected_tasks):

            n = batch_size // n_groups

            if group_index < batch_size % n_groups:
                n += 1

            class_keys = sorted([
                k for k in self.items
                if k[0] == task and self.items[k]
            ])

            for j in range(n):
                key = class_keys[j % len(class_keys)]
                exemplar = random.choice(self.items[key])
                samples.append((key, exemplar))

        random.shuffle(samples)
        return samples


# ======================== LOSS FUNCTIONS =======================

def replay_losses(model, memory, train_data, method):

    zero = next(model.parameters()).new_zeros(())

    result = {
        "replay": zero,
        "feature": zero,
        "prototype": zero,
        "distillation": zero
    }

    if len(memory) == 0:
        return result

    sample = memory.sample(REPLAY_BATCH)

    if not sample:
        return result

    indices = [
        exemplar["index"]
        for _, exemplar in sample
    ]

    images = np.array(
        train_data.data[indices], copy=True
    )

    x = torch.from_numpy(images).to(
        DEVICE
    ).permute(0, 3, 1, 2).float().div_(255.0)

    x = (x - MEAN) / STD

    total = len(sample)

    for task in sorted({key[0] for key, _ in sample}):

        positions = [
            i for i, (key, _) in enumerate(sample)
            if key[0] == task
        ]

        group = [sample[i] for i in positions]

        f, logits = model(
            x[positions], task, return_features=True
        )

        labels = torch.tensor(
            [key[1] for key, _ in group],
            device=DEVICE,
            dtype=torch.long
        )

        result["replay"] = result["replay"] + (
            F.cross_entropy(
                logits.float(), labels,
                reduction="sum"
            ) / total
        )

        if method == "replay_only":
            continue

        normalized = F.normalize(
            f.float(), dim=1
        )

        teacher_features = torch.stack([
            item["feature"]
            for _, item in group
        ]).to(DEVICE).float()

        prototypes = torch.stack([
            memory.prototypes[key]
            for key, _ in group
        ]).to(DEVICE).float()

        teacher_logits = torch.stack([
            item["logits"]
            for _, item in group
        ]).to(DEVICE).float()

        result["feature"] = result["feature"] + (
            (
                1.0 - F.cosine_similarity(
                    normalized,
                    teacher_features, dim=1
                )
            ).clamp_min(0).sum() / total
        )

        result["prototype"] = result["prototype"] + (
            (
                1.0 - F.cosine_similarity(
                    normalized, prototypes, dim=1
                )
            ).clamp_min(0).sum() / total
        )

        teacher_probability = F.softmax(
            teacher_logits / TEMPERATURE, dim=1
        )

        student_log_probability = F.log_softmax(
            logits.float() / TEMPERATURE, dim=1
        )

        kl = F.kl_div(
            student_log_probability,
            teacher_probability,
            reduction="sum"
        ) * TEMPERATURE**2

        result["distillation"] = (
            result["distillation"] + kl / total
        )

    return result


# ======================== EVALUATION ===========================

@torch.no_grad()
def evaluate(model, test_data, task):

    model.eval()

    loader = make_loader(test_data, task)

    correct = 0
    count = 0

    tp = np.zeros(5)
    predicted = np.zeros(5)
    actual = np.zeros(5)

    for x, y, _ in loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        with torch.autocast(
            "cuda", enabled=AMP
        ):
            outputs = model(x, task)
            prediction = outputs.argmax(1)

        correct += int((prediction == y).sum())
        count += len(y)

        for c in range(5):
            tp[c] += int(
                ((prediction == c) & (y == c)).sum()
            )
            predicted[c] += int(
                (prediction == c).sum()
            )
            actual[c] += int((y == c).sum())

    assert count == 500

    f1 = np.mean([
        2 * tp[c] / max(1, predicted[c] + actual[c])
        for c in range(5)
    ])

    return correct / count, float(f1)


def compute_metrics(matrix, task):

    values = matrix[task, :task + 1]

    accuracy = float(np.nanmean(values))

    if task == 0:
        return accuracy, 0.0, 0.0

    forgetting = []
    backward_transfer = []

    for j in range(task):

        best_previous = np.nanmax(
            matrix[j:task + 1, j]
        )

        forgetting.append(
            max(0.0, best_previous - matrix[task, j])
        )

        backward_transfer.append(
            matrix[task, j] - matrix[j, j]
        )

    return (
        accuracy,
        float(np.mean(forgetting)),
        float(np.mean(backward_transfer))
    )


# ======================== CHECKPOINT RECOVERY ===================

def run_folder(method, seed):
    return OUT / method / f"seed_{seed}"


def load_checkpoint(path):
    try:
        return torch.load(
            path, map_location="cpu",
            weights_only=False
        )
    except TypeError:
        return torch.load(path, map_location="cpu")


def locate_checkpoint(method, seed):

    local = run_folder(method, seed) / "checkpoint_after_task.pt"

    candidates = []

    if local.exists():
        candidates.append(local)

    source = Path("/kaggle/input")

    if source.exists():
        for path in source.rglob("checkpoint_after_task.pt"):
            if (
                path.parent.name == f"seed_{seed}"
                and path.parent.parent.name == method
            ):
                candidates.append(path)

    best = None

    for path in candidates:
        try:
            state = load_checkpoint(path)

            if state.get("fingerprint") != FINGERPRINT:
                continue

            if state.get("method") != method:
                continue

            if int(state.get("seed", -1)) != seed:
                continue

            if (
                best is None
                or state["next_task"] > best["next_task"]
            ):
                best = state

        except Exception as e:
            log("Checkpoint warning:", str(e))

    return best


def import_previous_metrics():

    source = Path("/kaggle/input")

    if not source.exists():
        return

    for path in source.rglob("task_metrics.csv"):

        try:
            method = path.parent.parent.name
            seed = int(path.parent.name.split("_")[-1])

            if method not in METHODS or seed not in SEEDS:
                continue

            config_path = path.parent / "run_configuration.json"

            if not config_path.exists():
                continue

            metadata = json.loads(
                config_path.read_text()
            )

            if metadata.get("fingerprint") != FINGERPRINT:
                continue

            destination = run_folder(method, seed)
            destination.mkdir(parents=True, exist_ok=True)

            for name in [
                "task_metrics.csv",
                "accuracy_matrix.csv",
                "f1_matrix.csv",
                "run_configuration.json"
            ]:
                old = path.parent / name
                new = destination / name

                if old.exists() and not new.exists():
                    shutil.copy2(old, new)

        except Exception:
            continue


# ======================== RESULTS COLLECTION ====================

def collect_runs():

    records = []
    curves = {}

    for method in METHODS:
        for seed in SEEDS:

            path = run_folder(method, seed) / "task_metrics.csv"

            if not path.exists():
                continue

            try:
                df = pd.read_csv(path)

                if df.empty:
                    continue

                df = df.sort_values("tasks_learned")

                tasks_completed = len(df)

                valid = (
                    tasks_completed <= TASKS
                    and df["tasks_learned"].tolist()
                    == list(range(1, tasks_completed + 1))
                )

                if not valid:
                    log("Invalid metrics:", path)
                    continue

                last = df.iloc[-1]

                records.append({
                    "method": method,
                    "seed": seed,
                    "tasks": tasks_completed,
                    "status": (
                        "completed"
                        if tasks_completed == TASKS
                        else "partial"
                    ),
                    "accuracy": float(
                        last["average_accuracy"]
                    ) * 100,
                    "forgetting": float(
                        last["average_forgetting"]
                    ) * 100,
                    "bwt": float(
                        last["backward_transfer"]
                    ) * 100,
                    "f1": float(
                        last["average_macro_f1"]
                    ) * 100,
                    "training_seconds": float(
                        df["task_seconds"].sum()
                    )
                })

                curves[(method, seed)] = df

            except Exception as e:
                log("Result read warning:", str(e))

    return pd.DataFrame(records), curves


# ======================== PDF GENERATION ========================

def pdf_text(pdf, title, lines):

    expanded = []

    for line in lines:
        if not str(line):
            expanded.append("")
        else:
            expanded.extend(
                textwrap.wrap(
                    str(line),
                    width=100,
                    break_long_words=True
                ) or [""]
            )

    page_size = 43

    for start in range(0, max(1, len(expanded)), page_size):

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")

        fig.text(
            0.07, 0.955,
            title,
            fontsize=16,
            fontweight="bold",
            va="top"
        )

        y = 0.91

        for line in expanded[start:start + page_size]:

            fig.text(
                0.07, y, line,
                fontsize=8.5,
                family="monospace",
                va="top"
            )

            y -= 0.019

        pdf.savefig(fig)
        plt.close(fig)


def plot_curves(pdf, curves, column, title, ylabel):

    fig, ax = plt.subplots(figsize=(9, 5.8))

    plotted = False

    for method in METHODS:

        trajectories = []

        for seed in SEEDS:

            df = curves.get((method, seed))

            if df is not None and len(df) == TASKS:
                trajectories.append(
                    df[column].to_numpy(dtype=float) * 100
                )

        if not trajectories:
            continue

        values = np.stack(trajectories)
        x = np.arange(1, TASKS + 1)

        mean = values.mean(axis=0)

        ax.plot(
            x, mean,
            marker="o",
            markersize=3,
            label=f"{method} (n={len(values)})"
        )

        if len(values) > 1:
            sd = values.std(axis=0, ddof=1)

            ax.fill_between(
                x,
                mean - sd,
                mean + sd,
                alpha=0.15
            )

        plotted = True

    if plotted:
        ax.set_title(title)
        ax.set_xlabel("Number of tasks learned")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(1, 21))
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)

    plt.close(fig)


def create_report(status="in_progress"):

    runs, curves = collect_runs()

    completed = (
        runs[runs.status == "completed"].copy()
        if not runs.empty else pd.DataFrame()
    )

    if not runs.empty:
        atomic_csv(runs, OUT / "all_runs.csv")

    summaries = []

    if not completed.empty:

        for method, group in completed.groupby("method"):

            summaries.append({
                "method": method,
                "seeds_completed": len(group),
                "accuracy_mean": group.accuracy.mean(),
                "accuracy_sd": group.accuracy.std(ddof=1),
                "forgetting_mean": group.forgetting.mean(),
                "forgetting_sd": group.forgetting.std(ddof=1),
                "bwt_mean": group.bwt.mean(),
                "f1_mean": group.f1.mean()
            })

    summary = pd.DataFrame(summaries)

    if not summary.empty:
        atomic_csv(summary, OUT / "method_summary.csv")

    complete_pairs = set()

    if not completed.empty:
        complete_pairs = set(
            zip(completed.method, completed.seed)
        )

    schedule = [
        (method, seed)
        for method in METHODS
        for seed in SEEDS
    ]

    remaining = [
        (m, s) for m, s in schedule
        if (m, s) not in complete_pairs
    ]

    fully_complete = not remaining

    atomic_json({
        "status": (
            "complete"
            if fully_complete else status
        ),
        "completed_runs": len(complete_pairs),
        "planned_runs": len(schedule),
        "remaining": remaining,
        "pdf": str(PDF),
        "full_study_completed": fully_complete
    }, OUT / "STATUS.json")

    lines = [
        "PROTOTYPE-GUIDED CONTINUAL LEARNING",
        "",
        "Original CIFAR-100: 50,000 train / 10,000 test.",
        "20 sequential tasks, five classes per task.",
        "Task-incremental evaluation: task ID known at inference.",
        "ResNet-18-style network with task-specific normalization.",
        "Ablations: one added loss term removed,",
        "with all other method settings retained.",
        "Each ablation removes exactly one added loss term.",
        "",
        f"GPU: {torch.cuda.get_device_name(0)}",
        f"Configured seeds: {SEEDS}",
        f"Epochs per task: {EPOCHS}",
        f"Memory budget: {MEMORY_BUDGET} images",
        "",
        f"Execution status: {status}",
        f"Fully completed experiments: {len(complete_pairs)}/15",
        "",
        "PREVIOUS FULL PGRFA REFERENCE:",
        "Average accuracy: 82.008%",
        "Average forgetting: 7.116%",
        "",
        "Full PGRFA reference comes from the earlier study.",
        "Ablation comparisons assess component contributions.",
        ""
    ]

    if not summary.empty:

        for _, row in summary.iterrows():

            lines.extend([
                f"METHOD: {row.method}",
                f"Completed seeds: {int(row.seeds_completed)}",
                f"Mean accuracy: {row.accuracy_mean:.3f}%",
                f"Accuracy SD: {row.accuracy_sd:.3f} pp",
                f"Mean forgetting: {row.forgetting_mean:.3f}%",
                f"Mean backward transfer: {row.bwt_mean:.3f} pp",
                f"Mean macro-F1: {row.f1_mean:.3f}%",
                ""
            ])

    if not completed.empty:

        imp = completed[
            completed.method == "improved"
        ].set_index("seed")

        base = completed[
            completed.method == "replay_only"
        ].set_index("seed")

        shared_seeds = sorted(
            set(imp.index) & set(base.index)
        )

        if shared_seeds:

            gains = [
                imp.loc[s, "accuracy"]
                - base.loc[s, "accuracy"]
                for s in shared_seeds
            ]

            reductions = [
                base.loc[s, "forgetting"]
                - imp.loc[s, "forgetting"]
                for s in shared_seeds
            ]

            lines.extend([
                "MATCHED REPLAY-ONLY COMPARISON:",
                f"Paired seeds: {len(shared_seeds)}",
                f"Mean accuracy gain: {np.mean(gains):+.3f} pp",
                f"Mean forgetting reduction: {np.mean(reductions):+.3f} pp",
                ""
            ])

            atomic_json({
                "paired_seeds": shared_seeds,
                "accuracy_gains_pp": gains,
                "forgetting_reductions_pp": reductions,
                "mean_accuracy_gain_pp": float(np.mean(gains)),
                "mean_forgetting_reduction_pp": float(
                    np.mean(reductions)
                )
            }, OUT / "paired_comparison.json")

    if remaining:
        lines.extend([
            "REMAINING EXPERIMENTS:",
            *[
                f"{method}, seed {seed}"
                for method, seed in remaining
            ]
        ])

    temporary_pdf = WORK / "PGRFA_ABLATION_RESULTS.pdf.temp"

    with PdfPages(temporary_pdf) as pdf:

        pdf_text(
            pdf,
            "Final PGRFA Research Results",
            lines
        )

        if not runs.empty:

            table_lines = [
                "METHOD | SEED | TASKS | ACC% | FORGET% | BWT% | F1%",
                "-" * 77
            ]

            for _, row in runs.iterrows():

                table_lines.append(
                    f"{row.method:<13} "
                    f"{int(row.seed):>4} "
                    f"{int(row.tasks):>3}/20 "
                    f"{row.accuracy:>7.2f} "
                    f"{row.forgetting:>8.2f} "
                    f"{row.bwt:>7.2f} "
                    f"{row.f1:>7.2f}"
                )

            pdf_text(
                pdf,
                "Individual Seed Results",
                table_lines
            )

        plot_curves(
            pdf, curves,
            "average_accuracy",
            "Average Accuracy Across Tasks",
            "Accuracy (%)"
        )

        plot_curves(
            pdf, curves,
            "average_forgetting",
            "Forgetting Across Tasks",
            "Forgetting (percentage points)"
        )

        plot_curves(
            pdf, curves,
            "backward_transfer",
            "Backward Transfer Across Tasks",
            "Backward transfer (percentage points)"
        )

        if not summary.empty:

            fig, ax = plt.subplots(figsize=(9, 5.8))

            ax.bar(
                summary.method,
                summary.accuracy_mean
            )

            ax.axhline(
                PUBLISHED_ACC,
                linestyle="--",
                color="gray",
                label="Previous full PGRFA (5 seeds)"
            )

            ax.set_ylabel("Average accuracy (%)")
            ax.set_title("Final Accuracy Comparison")
            ax.legend()
            ax.grid(axis="y", alpha=0.2)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # Individual-task accuracy matrix.
        matrix_path = (
            run_folder("no_feature", SEEDS[0])
            / "accuracy_matrix.csv"
        )

        if matrix_path.exists():

            try:
                matrix = pd.read_csv(
                    matrix_path
                ).to_numpy(dtype=float)

                fig, ax = plt.subplots(figsize=(9, 7))

                image = ax.imshow(
                    np.ma.masked_invalid(matrix * 100),
                    vmin=0, vmax=100,
                    aspect="auto"
                )

                ax.set_title(
                    "Task Accuracy Matrix - No Feature, Seed 42"
                )
                ax.set_xlabel("Task evaluated")
                ax.set_ylabel("Tasks learned")

                fig.colorbar(
                    image, ax=ax,
                    label="Accuracy (%)"
                )

                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            except Exception as e:
                log("Matrix plot warning:", str(e))

        pdf_text(
            pdf,
            "Scientific Interpretation",
            [
                "All numerical results originate from saved",
                "experimental metric files.",
                "",
                "Only complete 20-task runs contribute to",
                "method-level final averages.",
                "",
                "Five independent seeds are configured for",
                "the improved and matched baseline methods.",
                "",
                "The published benchmark and the current",
                "method are not identical experimental protocols.",
                "",
                "Task-incremental accuracy uses known task IDs.",
                "It must not be described as unrestricted",
                "100-class class-incremental accuracy.",
                "",
                "GPU training does not guarantee that a",
                "proposed method will outperform a paper.",
                "",
                "Conclusions require measured accuracy,",
                "forgetting, matched controls, and reproducibility."
            ]
        )

    os.replace(temporary_pdf, PDF)

    shutil.copy2(
        PDF, OUT / "PGRFA_ABLATION_RESULTS.pdf"
    )

    temp_zip = WORK / "PGRFA_ABLATION_RESULTS.zip.temp"

    with zipfile.ZipFile(
        temp_zip,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:

        for path in OUT.rglob("*"):

            if not path.is_file():
                continue

            if path.suffix.lower() not in {
                ".csv", ".json", ".pdf", ".txt"
            }:
                continue

            archive.write(
                path,
                arcname=str(path.relative_to(OUT))
            )

    os.replace(temp_zip, ZIP)

    log(
        "PDF UPDATED:",
        PDF,
        "| Complete runs:",
        len(complete_pairs),
        "/15"
    )


# ======================== FULL TRAINING =========================

def train_one_seed(method, seed, train_data, test_data):

    folder = run_folder(method, seed)
    folder.mkdir(parents=True, exist_ok=True)

    existing_metrics = folder / "task_metrics.csv"

    if existing_metrics.exists():

        try:
            df = pd.read_csv(existing_metrics)

            if (
                len(df) == TASKS
                and df["tasks_learned"].tolist()
                == list(range(1, TASKS + 1))
            ):
                log("ALREADY COMPLETED:", method, seed)
                return True

        except Exception:
            pass

    seed_all(seed)

    model = ContinualResNet().to(DEVICE)
    memory = PrototypeMemory()

    accuracy_matrix = np.full(
        (TASKS, TASKS), np.nan
    )

    f1_matrix = np.full(
        (TASKS, TASKS), np.nan
    )

    rows = []
    starting_task = 0

    saved = locate_checkpoint(method, seed)

    if saved is not None:

        model.load_state_dict(saved["model"])
        memory.load_state_dict(saved["memory"])

        accuracy_matrix = saved["accuracy_matrix"]
        f1_matrix = saved["f1_matrix"]
        rows = saved["rows"]

        starting_task = int(saved["next_task"])

        if "python_rng" in saved:
            random.setstate(saved["python_rng"])

        if "numpy_rng" in saved:
            np.random.set_state(saved["numpy_rng"])

        if "torch_rng" in saved:
            torch.set_rng_state(saved["torch_rng"])

        if saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])

        log(
            "CHECKPOINT RESTORED:",
            method, seed,
            "starting task", starting_task + 1
        )

    atomic_json({
        "method": method,
        "seed": seed,
        "fingerprint": FINGERPRINT,
        "dataset": "Original CIFAR-100",
        "training_images": 50000,
        "test_images": 10000,
        "pretrained_weights": False,
        "evaluation": "Task-incremental"
    }, folder / "run_configuration.json")

    for task in range(starting_task, TASKS):

        if time.monotonic() >= DEADLINE:
            log("Session time budget reached.")
            return False

        log(
            "STARTING TASK", task + 1,
            "/", TASKS,
            "| METHOD", method,
            "| SEED", seed
        )

        model.set_training_task(task)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS,
            eta_min=MIN_LR
        )

        scaler = torch.amp.GradScaler(
            "cuda", enabled=AMP
        )

        task_start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()

        last_losses = {}

        for epoch in range(EPOCHS):

            model.set_training_task(task)

            # Deterministic task/epoch shuffle seed.
            loader = make_loader(
                train_data,
                task,
                augment=True,
                shuffle=True,
                seed=(
                    seed * 100000
                    + task * 1000
                    + epoch
                )
            )

            running = defaultdict(float)
            batches = 0

            for x, y, _ in loader:

                x = x.to(
                    DEVICE, non_blocking=True
                )

                y = y.to(
                    DEVICE, non_blocking=True
                )

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    "cuda",
                    dtype=torch.float16,
                    enabled=AMP
                ):

                    logits = model(x, task)

                    current_loss = F.cross_entropy(
                        logits.float(),
                        y,
                        label_smoothing=LABEL_SMOOTHING
                    )

                    old = replay_losses(
                        model,
                        memory,
                        train_data,
                        method
                    )

                    total = (
                        current_loss
                        + REPLAY_WEIGHT * old["replay"]
                    )

                    if method in METHODS:

                        total = (
                            total
                            + (0.0 if method == "no_feature" else FEATURE_WEIGHT) * old["feature"]
                            + (0.0 if method == "no_prototype" else PROTOTYPE_WEIGHT) * old["prototype"]
                            + (0.0 if method == "no_logit" else DISTILL_WEIGHT) * old["distillation"]
                        )

                if not torch.isfinite(total):
                    raise FloatingPointError(
                        f"Nonfinite loss: "
                        f"{method}, seed={seed}, "
                        f"task={task}, epoch={epoch}"
                    )

                scaler.scale(total).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5.0
                )

                scaler.step(optimizer)
                scaler.update()

                running["total"] += float(
                    total.detach().float()
                )

                running["current"] += float(
                    current_loss.detach().float()
                )

                for key in old:
                    running[key] += float(
                        old[key].detach().float()
                    )

                batches += 1

            scheduler.step()

            last_losses = {
                key: value / max(1, batches)
                for key, value in running.items()
            }

            if (
                epoch == 0
                or (epoch + 1) % 3 == 0
                or epoch + 1 == EPOCHS
            ):
                log(
                    "EPOCH", epoch + 1,
                    "/", EPOCHS,
                    "TASK", task + 1,
                    "METHOD", method,
                    "SEED", seed,
                    "LOSS",
                    round(last_losses["total"], 5)
                )

        stored = memory.update(
            model, train_data, task
        )

        task_accuracies = []
        task_f1s = []

        for previous in range(task + 1):

            accuracy, macro_f1 = evaluate(
                model, test_data, previous
            )

            task_accuracies.append(accuracy)
            task_f1s.append(macro_f1)

        accuracy_matrix[
            task, :task + 1
        ] = task_accuracies

        f1_matrix[
            task, :task + 1
        ] = task_f1s

        accuracy, forgetting, bwt = compute_metrics(
            accuracy_matrix, task
        )

        elapsed = time.monotonic() - task_start

        row = {
            "method": method,
            "seed": seed,
            "task_index": task,
            "tasks_learned": task + 1,
            "average_accuracy": accuracy,
            "average_forgetting": forgetting,
            "backward_transfer": bwt,
            "average_macro_f1": float(
                np.mean(task_f1s)
            ),
            "new_task_accuracy": task_accuracies[-1],
            "memory_images": stored,
            "task_seconds": elapsed,
            "peak_gpu_mb": (
                torch.cuda.max_memory_allocated()
                / (1024**2)
            )
        }

        for key, value in last_losses.items():
            row["last_epoch_" + key] = value

        rows.append(row)

        atomic_csv(
            pd.DataFrame(rows),
            folder / "task_metrics.csv"
        )

        atomic_csv(
            pd.DataFrame(accuracy_matrix),
            folder / "accuracy_matrix.csv"
        )

        atomic_csv(
            pd.DataFrame(f1_matrix),
            folder / "f1_matrix.csv"
        )

        state = {
            "fingerprint": FINGERPRINT,
            "method": method,
            "seed": seed,
            "next_task": task + 1,
            "model": {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            },
            "memory": memory.state_dict(),
            "accuracy_matrix": accuracy_matrix,
            "f1_matrix": f1_matrix,
            "rows": rows,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
        }

        atomic_checkpoint(
            state,
            folder / "checkpoint_after_task.pt"
        )

        log(
            "TASK COMPLETED:",
            method,
            "seed", seed,
            "tasks", task + 1,
            "accuracy", round(accuracy * 100, 3),
            "forgetting", round(forgetting * 100, 3),
            "BWT", round(bwt * 100, 3),
            "stored images", stored
        )

        # Generate an intermediate PDF every five tasks.
        if (task + 1) % 5 == 0:
            try:
                create_report("in_progress")
            except Exception as e:
                log("Nonfatal report warning:", repr(e))

        gc.collect()

    log(
        "FULL RUN COMPLETED:",
        method, seed,
        "final accuracy:",
        round(accuracy * 100, 3)
    )

    return True


# ======================== EXPERIMENT EXECUTION ===================

import_previous_metrics()

run_status = "in_progress"

try:

    TRAIN, TEST = prepare_cifar100()

    # First all five improved runs.
    # Then all five matched replay-only runs.
    for method in METHODS:

        for seed in SEEDS:

            if time.monotonic() >= DEADLINE:
                run_status = "time_budget_stop"
                break

            success = train_one_seed(
                method, seed, TRAIN, TEST
            )

            create_report("in_progress")

            gc.collect()
            torch.cuda.empty_cache()

            if not success:
                run_status = "time_budget_stop"
                break

        if run_status == "time_budget_stop":
            break

    else:
        run_status = "complete"

except Exception:

    run_status = "error_with_checkpoints"

    error_text = traceback.format_exc()

    (OUT / "ERROR_TRACEBACK.txt").write_text(
        error_text, encoding="utf-8"
    )

    print(error_text, flush=True)

finally:

    # Always attempt to produce the final available-results PDF.
    try:
        create_report(run_status)
    except Exception as report_error:
        print(
            "PDF generation error:",
            repr(report_error),
            flush=True
        )

# ======================== FINAL DOWNLOAD ========================

print("\n" + "=" * 70)
print("FINAL EXPERIMENT EXECUTION FINISHED")
print("=" * 70)

print("Status:", run_status)
print("PDF:", PDF)
print("Results ZIP:", ZIP)
print("Research directory:", OUT)

runs, _ = collect_runs()

if not runs.empty:

    print("\nAVAILABLE RESULTS")
    print(
        runs.round(3).to_string(index=False)
    )

    complete = runs[runs.status == "completed"]

    if not complete.empty:

        print("\nCOMPLETED RUN SUMMARY")

        print(
            complete.groupby("method")[
                ["accuracy", "forgetting", "bwt", "f1"]
            ].agg(["mean", "std"]).round(3)
        )

if PDF.exists():

    print(
        "\nPDF size:",
        round(PDF.stat().st_size / 1e6, 2),
        "MB"
    )

    try:
        from IPython.display import display, FileLink
        display(FileLink(str(PDF)))

        if ZIP.exists():
            display(FileLink(str(ZIP)))

    except Exception:
        pass

print("\nAll available results have been saved.")
print("A 'complete' study requires 15 completed 20-task runs.")
