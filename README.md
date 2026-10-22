# ESS-Flow: Training-free guidance as Bayesian inference in source space

The sampler in [`essflow/sampler.py`](essflow/sampler.py) runs parallel MCMC chains and only calls the potential for those chains that are still running.

```python
import torch
from essflow import EllipticalSliceSampler

def log_potential(z):            # z: [n, ...] source samples, n varies between calls
    x = transport_map(z)         # e.g. an ODE solve of your flow model
    return log_g(x)              # [n]

sampler = EllipticalSliceSampler(log_potential, max_subiter=100)
samples, nfe = sampler.sample(torch.randn(num_chains, *source_shape),
                              num_samples=1, num_warmup=200)
# samples: [num_chains, num_samples, *source_shape], decode them with transport_map
```

## Repository structure

```
essflow/              ESS sampler
experiments/
  materials/          FlowMM: ESS-Flow, unconditional, D-Flow, PnP-Flow, DAPS,
                      multi-fidelity reweighting, S.U.N.T. rate, property statistics
  proteins/           Proteina: ESS-Flow, unconditional, designability evaluation
  molecules/          GeoLDM: ESS-Flow, unconditional (DDPM/DDIM), evaluation
priors/               generative priors as git submodules
envs/                 conda environments (one per prior)
scripts/slurm/        slurm scripts with the configurations used in the paper
```

The priors are included as git submodules. For FlowMM and GeoLDM we use forks with minor additions:

- [FlowMM fork](https://github.com/Adhithyan8/flowmm): ODE solvers without `torch.no_grad` (for D-Flow) and integration from an intermediate time t (for PnP-Flow and DAPS).
- [GeoLDM fork](https://github.com/Adhithyan8/GeoLDM): a deterministic DDIM sampler from a given source point.

## Installation

Clone the repository with its submodules.

```bash
git clone https://github.com/Adhithyan8/essflow.git
cd essflow
git -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init --recursive
```

The three priors need different dependencies, so there is one conda environment per experiment. All commands are run from the repository root.

**Materials**

```bash
mamba env create -f envs/materials.yml
mamba activate essflow-materials
pip install --no-deps alignn==2025.4.1   # its dependencies are in the env file
pip install -e priors/flowmm/remote/cdvae -e priors/flowmm/remote/DiffCSP-official -e priors/flowmm
# manifm has no __init__.py, so it needs the path-based editable install
pip install -e priors/flowmm/remote/riemannian-fm --config-settings editable_mode=compat
(cd priors/flowmm && bash create_env_file.sh)
```

**Proteins**

```bash
mamba env create -f envs/proteins.yml
mamba activate essflow-proteins
pip install -e priors/proteina
(cd priors/proteina && bash script_utils/download_pmpnn_weghts.sh)   # ProteinMPNN weights
```

**Molecules**

```bash
mamba env create -f envs/molecules.yml
```

Our experiments were run with CUDA 12.1 on NVIDIA A100 GPUs.

## Pretrained models and data

Place the files as follows (`pretrained/` and `data/` are ignored by git):

```
pretrained/
  flowmm_mp20/                          FlowMM trained on MP-20
    .hydra/config.yaml
    checkpoints/flowmm_mp20.ckpt
  alignn/                               ALIGNN property predictors
    jv_bulk_modulus_kv_alignn.zip
    jv_shear_modulus_gv_alignn.zip
    jv_mbj_bandgap_alignn.zip
    jv_ehull_alignn.zip
  geoldm_drugs/                         GeoLDM trained on GEOM-Drugs
    args.pickle
    generative_model_ema.npy
  proteina_v1.3_DFS_60M_notri.ckpt      Proteina, 60M parameters
data/
  2023-02-07-ppd-mp.pkl                 Materials Project convex hull (for S.U.N.T.)
  proteina/                             Proteina auxiliary files
```

| File | Source |
|---|---|
| `pretrained/flowmm_mp20` | No FlowMM weights have been released, so we provide the checkpoint we trained on [Hugging Face](https://huggingface.co/Adhi94/essflow-flowmm-mp20): `hf download Adhi94/essflow-flowmm-mp20 --local-dir pretrained/flowmm_mp20` (the `hf` command comes with `pip install huggingface_hub`). It was trained in `priors/flowmm` with `python scripts_model/run.py data=mp_20 model=abits_params` (see the FlowMM README). |
| `pretrained/alignn` | Pretrained JARVIS-DFT models from [ALIGNN](https://figshare.com/projects/ALIGNN_models/126478): [bulk modulus](https://figshare.com/ndownloader/files/31458649), [shear modulus](https://figshare.com/ndownloader/files/31458724), [band gap](https://figshare.com/ndownloader/files/31458694), [energy above hull](https://figshare.com/ndownloader/files/31458658). Download them in a browser, since figshare can block scripted downloads, and save them under the names above. |
| `pretrained/geoldm_drugs` | The GEOM-Drugs model from the [GeoLDM pretrained models](https://drive.google.com/drive/folders/1EQ9koVx-GA98kaKBS8MZ_jJ8g4YhdKsL) (`latent_nf=2`). |
| `pretrained/proteina_v1.3_DFS_60M_notri.ckpt` | [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/resources/proteina_v1.3_dfs_60m_notri/files) |
| `data/proteina` | Contents of [proteina_additional_files.zip](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/resources/proteina_additional_files/files) (another location can be set with `export DATA_PATH=...`). |
| `data/2023-02-07-ppd-mp.pkl` | Convex hull from [Matbench Discovery](https://figshare.com/articles/dataset/Matbench_Discovery_v1_0_0/22715158?file=40344451), unzipped. Only needed for the S.U.N.T. rates. |

ESMFold and CHGNet are downloaded automatically on first use. ESMFold is cached by Hugging Face (set `HF_HOME` to change the location).

## Running the experiments

Each script can be run directly with `python -m experiments.<domain>.<script> --help`. The slurm scripts in `scripts/slurm/` contain the configurations used in the paper and must be submitted from the repository root. [`scripts/slurm/activate.sh`](scripts/slurm/activate.sh) loads mamba with a module on the Berzelius cluster, so adapt it to how mamba is set up on your cluster, and add your account with `sbatch -A <account>` if needed. Outputs are written to `results/`.

### Materials

Tasks: `bulk_modulus`, `shear_modulus`, `band_gap`, `e_hull` (stability), `space_group` (P6<sub>3</sub>/mmc).

```bash
TASK=bulk_modulus

# ESS-Flow: 8 array jobs x 125 chains = 1000 samples
sbatch scripts/slurm/materials/essflow.sh $TASK

# unconditional samples and the gradient-based baselines (not for space_group),
for METHOD in unconditional dflow pnpflow daps; do
    sbatch scripts/slurm/materials/baseline.sh $METHOD $TASK
done

# multi-fidelity: reweight the ESS-Flow samples (50 ODE steps) to 1000 ODE steps
sbatch scripts/slurm/materials/reweight.sh $TASK
```

When the samples are done, compute the property errors, histograms and the S.U.N.T. rates:

```bash
python -m experiments.materials.analyze --task $TASK
for METHOD in unconditional dflow pnpflow daps essflow reweighted; do
    sbatch scripts/slurm/materials/sunt.sh $METHOD $TASK
done
```

`analyze` prints the mean absolute error to the target (the mean predicted energy above hull for `e_hull`, the match rate for `space_group`) and the number of function evaluations of ESS-Flow. It also saves the property histograms to `results/materials/<task>/histogram.pdf`. Each S.U.N.T. job prints the valid, S., U.N., T., S.U.N. and S.U.N.T. rates and saves them to `results/materials/<task>/sunt_<method>.json`.

### Protein backbones

```bash
for W in 0.5 1.0 2.0; do sbatch scripts/slurm/proteins/essflow.sh $W; done
sbatch scripts/slurm/proteins/unconditional.sh

python -m experiments.proteins.evaluate results/proteins/unconditional.pt results/proteins/essflow_w*.pt
```

ESS-Flow runs 16 chains for 264 steps and keeps the last 64 states of each chain, giving 1024 samples. The evaluation prints the mean scRMSD, the designability and the mean pairwise TM-score among the designable samples.

### Molecules

```bash
for W in 5 10 20; do sbatch scripts/slurm/molecules/essflow.sh $W; done
sbatch scripts/slurm/molecules/unconditional.sh ddpm
sbatch scripts/slurm/molecules/unconditional.sh ddim

python -m experiments.molecules.evaluate results/molecules/unconditional_*.pt results/molecules/essflow_w*.pt
```

The evaluation prints connectivity, diversity, uniqueness, QED, SA and quality.
