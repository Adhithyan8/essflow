"""Self-consistency RMSD (scRMSD) of protein backbones.

For each backbone, ``num_seq`` sequences are designed with ProteinMPNN (CA-only
model, temperature 0.1) and refolded with ESMFold. The scRMSD is the minimum
CA RMSD between the backbone and the refolded structures after alignment.
Both models are loaded once and called in batches.
"""

import copy
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from experiments.proteins.proteina import PROTEINA_DIR


class ScRMSDOracle:
    def __init__(self, num_seq=1, temp=0.1, device=None, cache_dir=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_seq = num_seq
        self.temp = temp
        self._call_count = 0
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="scrmsd_"))

        # ProteinMPNN (CA-only weights)
        mpnn_dir = PROTEINA_DIR / "ProteinMPNN"
        if str(mpnn_dir) not in sys.path:
            sys.path.insert(0, str(mpnn_dir))
        from protein_mpnn_utils import (
            ProteinMPNN,
            StructureDatasetPDB,
            _S_to_seq,
            parse_PDB,
            tied_featurize,
        )

        ckpt = torch.load(
            str(mpnn_dir / "ca_model_weights" / "v_48_020.pt"), map_location=self.device
        )
        self._mpnn = ProteinMPNN(
            ca_only=True, num_letters=21,
            node_features=128, edge_features=128, hidden_dim=128,
            num_encoder_layers=3, num_decoder_layers=3,
            augment_eps=0.0, k_neighbors=ckpt["num_edges"],
        ).to(self.device).eval()
        self._mpnn.load_state_dict(ckpt["model_state_dict"])
        self._parse_PDB = parse_PDB
        self._StructureDatasetPDB = StructureDatasetPDB
        self._tied_featurize = tied_featurize
        self._S_to_seq = _S_to_seq
        self._omit_AAs_np = np.zeros(21, dtype=np.float32)
        self._bias_AAs_np = np.zeros(21, dtype=np.float32)

        # ESMFold
        from transformers import AutoTokenizer, EsmForProteinFolding
        from transformers import logging as hf_logging
        from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

        hf_logging.set_verbosity_error()
        # weights are downloaded from Hugging Face (cache location: $HF_HOME)
        self._esm_tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1", cache_dir=cache_dir)
        self._esm = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1", cache_dir=cache_dir
        ).to(self.device).eval()
        self._atom14_to_atom37 = atom14_to_atom37

        from proteinfoundation.metrics.designability import rmsd_metric
        from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb

        self._rmsd_metric = rmsd_metric
        self._write_prot_to_pdb = write_prot_to_pdb

    def _write_pdb(self, atom37_np):
        self._call_count += 1
        pdb_path = self._tmp_dir / f"sample_{self._call_count:06d}.pdb"
        self._write_prot_to_pdb(atom37_np, str(pdb_path), overwrite=True, no_indexing=True)
        # make sure the file is complete on network file systems before parsing
        fd = os.open(str(pdb_path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return pdb_path

    def _featurize_pdb(self, pdb_path):
        pdb_dict_list = self._parse_PDB(str(pdb_path), ca_only=True)
        name = pdb_dict_list[0]["name"]
        all_chains = [k[-1:] for k in pdb_dict_list[0] if k[:9] == "seq_chain"]
        chain_id_dict = {name: (["A"], [c for c in all_chains if c != "A"])}
        dataset = self._StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=20000)
        protein = next(iter(dataset))
        return self._tied_featurize(
            [copy.deepcopy(protein)], self.device, chain_id_dict,
            None, None, None, None, None, ca_only=True,
        )

    @torch.no_grad()
    def _mpnn_sample_batch(self, feats_list):
        """``num_seq`` ProteinMPNN sequences for each structure, in one batch."""

        def cat(idx):
            return torch.cat([f[idx] for f in feats_list], dim=0)

        # indices into the outputs of tied_featurize
        X, S, mask = cat(0), cat(1), cat(2)
        chain_M, chain_enc = cat(4), cat(5)
        chain_M_pos, omit_AA_mask, residue_idx = cat(10), cat(11), cat(12)
        pssm_coef, pssm_bias, pssm_log_odds_all = cat(15), cat(16), cat(17)
        bias_by_res_all = cat(18)
        pssm_log_odds_mask = (pssm_log_odds_all > 0.0).float()

        all_seqs = [[] for _ in feats_list]
        for _ in range(self.num_seq):
            randn = torch.randn(chain_M.shape, device=self.device)
            sample = self._mpnn.sample(
                X, randn, S, chain_M, chain_enc, residue_idx,
                mask=mask, temperature=self.temp,
                omit_AAs_np=self._omit_AAs_np,
                bias_AAs_np=self._bias_AAs_np,
                chain_M_pos=chain_M_pos,
                omit_AA_mask=omit_AA_mask,
                pssm_coef=pssm_coef, pssm_bias=pssm_bias,
                pssm_multi=0.0,
                pssm_log_odds_flag=False,
                pssm_log_odds_mask=pssm_log_odds_mask,
                pssm_bias_flag=False,
                bias_by_res=bias_by_res_all,
            )
            for i in range(len(feats_list)):
                all_seqs[i].append(self._S_to_seq(sample["S"][i], chain_M[i]))
        return all_seqs

    @torch.no_grad()
    def _esmfold_atom37(self, seqs, esm_batch_size=64):
        all_a37 = []
        for start in range(0, len(seqs), esm_batch_size):
            inputs = self._esm_tok(
                seqs[start : start + esm_batch_size],
                return_tensors="pt", add_special_tokens=False, padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self._esm(**inputs)
            a37 = self._atom14_to_atom37(outputs["positions"][-1], outputs)
            all_a37.append(a37.cpu().float())
        return torch.cat(all_a37, dim=0)

    def __call__(self, atom37_batch):
        """scRMSD of each structure in ``atom37_batch`` ``[B, n_res, 37, 3]``."""
        atom37_np = [a.cpu().numpy() for a in atom37_batch]
        feats_list = [self._featurize_pdb(self._write_pdb(a)) for a in atom37_np]
        seqs_per_struct = self._mpnn_sample_batch(feats_list)
        refolded = self._esmfold_atom37([s for seqs in seqs_per_struct for s in seqs])

        results, idx = [], 0
        for a, seqs in zip(atom37_np, seqs_per_struct):
            generated = torch.tensor(a, dtype=torch.float32)
            results.append(
                min(
                    float(self._rmsd_metric(generated, refolded[idx + j]))
                    for j in range(len(seqs))
                )
            )
            idx += len(seqs)
        return results
