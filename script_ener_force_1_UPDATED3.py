import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import math

# ============================================================
# Unit helpers
# ============================================================
HARTREE_TO_KCAL_MOL = 627.503
BOHR_TO_ANG = 0.529177
EH_PER_BOHR_TO_KCAL_PER_ANG = HARTREE_TO_KCAL_MOL / BOHR_TO_ANG  # ~1185.821


def convert_forces(forces: np.ndarray, from_units: str) -> np.ndarray:
    """Convert forces to kcal/mol/Å."""
    u = from_units.strip().lower().replace(" ", "")
    if u in ("kcal/mol/ang", "kcal/mol/a", "kcal/mol/å"):
        return forces
    if u in ("eh/bohr", "hartree/bohr", "hartree/boh"):
        return forces * EH_PER_BOHR_TO_KCAL_PER_ANG
    raise ValueError(f"Unknown force unit: {from_units}")


# ============================================================
# Parsers for LAMMPS dumps
# ============================================================
def _read_dump_after_atoms(file_path: str) -> List[str]:
    with open(file_path, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith("ITEM: ATOMS"):
            return lines[i + 1 :]
    raise ValueError("Could not find 'ITEM: ATOMS' in dump: %s" % file_path)


def read_sna_dump_add_c0(file_path: str) -> np.ndarray:
    """Read sna/atom and prepend C0=1.0 to each per-atom row."""
    atom_lines = _read_dump_after_atoms(file_path)
    data = []
    for line in atom_lines:
        toks = line.split()
        if not toks:
            continue
        floats = list(map(float, toks[1:]))  # skip atom id
        data.append([1.0] + floats)          # C0 column
    return np.asarray(data, dtype=float)     # shape (N_atoms, K)


def read_snad_dump(
    file_path: str,
    n_species: int,
    k_features_without_c0: Optional[int] = None
) -> np.ndarray:
    """
    Read snad/atom (derivatives) and reshape to (N_atoms, n_species, 3, K0),
    where K0 excludes the C0 column.
    """
    atom_lines = _read_dump_after_atoms(file_path)
    rows = []
    for line in atom_lines:
        toks = line.split()
        if not toks:
            continue
        vals = np.array(list(map(float, toks[1:])), dtype=float)
        rows.append(vals)
    if not rows:
        raise ValueError("No atom rows read from SNAD dump: %s" % file_path)

    rows = np.stack(rows, axis=0)
    n_atoms, ncols = rows.shape
    if k_features_without_c0 is None:
        if ncols % (n_species * 3) != 0:
            raise ValueError(
                f"SNAD columns {ncols} not divisible by n_species*3={n_species*3}. "
                "Please pass k_features_without_c0 explicitly or check dump."
            )
        k_features_without_c0 = ncols // (n_species * 3)

    expected = n_species * 3 * k_features_without_c0
    if ncols != expected:
        raise ValueError(
            f"Unexpected SNAD shape: got {ncols} cols, expected {expected} (= n_species*3*K0)."
        )
    dB = rows.reshape(n_atoms, n_species, 3, k_features_without_c0)  # no C0 yet
    return dB


def pad_snad_with_c0_zero(dB_no_c0: np.ndarray) -> np.ndarray:
    """Insert a leading zero column for C0 derivative: (N,S,3,K0) -> (N,S,3,K0+1)."""
    N, S, C, K0 = dB_no_c0.shape
    dB = np.zeros((N, S, C, K0 + 1), dtype=float)
    dB[..., 1:] = dB_no_c0  # C0 derivative stays zero
    return dB


# ============================================================
# Normalization
# ============================================================
@dataclass
class SnapNormalizer:
    means: Dict[str, np.ndarray]
    stds: Dict[str, np.ndarray]
    first_species: str

    @staticmethod
    def fit(
        all_B_by_struct: List[np.ndarray],
        all_species_by_struct: List[List[str]],
        species_list: List[str]
    ) -> "SnapNormalizer":
        means, stds = {}, {}
        for s in species_list:
            rows = []
            for B, atom_species in zip(all_B_by_struct, all_species_by_struct):
                for i, sp in enumerate(atom_species):
                    if sp == s:
                        rows.append(B[i])
            rows = np.array(rows, dtype=float)
            mu = rows.mean(axis=0)
            sd = rows.std(axis=0)
            sd[sd == 0.0] = 1.0
            means[s] = mu
            stds[s] = sd
        return SnapNormalizer(means=means, stds=stds, first_species=species_list[0])

    def normalize_B(self, B: np.ndarray, atom_species: List[str]) -> np.ndarray:
        """Z-score per species; restore raw C0 for the *first* species only."""
        out = B.copy()
        for i, sp in enumerate(atom_species):
            mu, sd = self.means[sp], self.stds[sp]
            out[i] = (B[i] - mu) / sd
            if sp == self.first_species:
                out[i, 0] = B[i, 0]  # keep C0 raw for the first species only
        return out

    def scale_snad(self, dB: np.ndarray, species_list: List[str]) -> np.ndarray:
        """
        Divide SNAD by per-species std so it matches normalized B;
        zero-out the (inserted) C0 derivative column explicitly.
        """
        N, S, C, K = dB.shape
        scaled = dB.copy()
        for s_idx, sp in enumerate(species_list):
            sd = self.stds[sp]
            scaled[:, s_idx, :, :] = scaled[:, s_idx, :, :] / sd  # broadcast over K
            scaled[:, s_idx, :, 0] = 0.0  # ensure exact zero for C0
        return scaled


# ============================================================
# Design matrices
# ============================================================
def energy_row_from_B(
    norm_B: np.ndarray,
    atom_species: List[str],
    species_list: List[str]
) -> np.ndarray:
    """Sum normalized per-atom bispectra by species -> one row (S*K,)."""
    S = len(species_list)
    K = norm_B.shape[1]
    species_index = {s: i for i, s in enumerate(species_list)}
    row = np.zeros((S, K), dtype=float)
    for i, sp in enumerate(atom_species):
        row[species_index[sp]] += norm_B[i]
    return row.reshape(-1)


def force_rows_from_snad(norm_snad: np.ndarray) -> np.ndarray:
    """Build Jacobian for forces: Xf shape (3N, S*K) with physics sign (minus)."""
    N, S, _, K = norm_snad.shape
    Xf = np.empty((3 * N, S * K), dtype=float)
    r = 0
    for i in range(N):
        for a in range(3):
            block = -norm_snad[i, :, a, :]  # F = - sum beta * dB/dR
            Xf[r] = block.reshape(-1)
            r += 1
    return Xf


# ============================================================
# Fit (energies + forces) with ONE C0 and NO intercept
# ============================================================
def fit_ridge_energy_forces(
    training_set: List[Dict],
    species_list: List[str],
    lambda_en: float = 0.1,
    w_energy_sq: float = 1.0,
    w_forces_sq: float = 1.0,
    lammps_type_order: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Ridge, SnapNormalizer, int]:
    """
    training_set entries must have:
      - 'bispec_file': sna dump
      - 'snad_file'  : snad dump (derivatives)
      - 'atom_species': list of species tags per atom
      - 'energy': total energy [kcal/mol]
      - 'forces': (N,3) array, *kcal/mol/Å* (or set 'force_units' to 'Eh/bohr' to auto-convert)
      - optional 'force_units': 'kcal/mol/Å' (default) or 'Eh/bohr'
    Returns:
      beta_matrix (S,K), model, normalizer, K
    """
    S = len(species_list)

    # Read all B (with C0) and all SNAD (pad C0 derivative), convert forces
    all_B, all_species_lists = [], []
    all_dB, all_forces, energies = [], [], []

    for entry in training_set:
        atom_species = entry["atom_species"]
        energies.append(float(entry["energy"]))

        # sna with C0
        B = read_sna_dump_add_c0(entry["bispec_file"])  # shape (N,K_with_c0)
        N, K_with_c0 = B.shape

        # snad without C0 -> pad zero column
        dB_no_c0 = read_snad_dump(entry["snad_file"], n_species=S)
        # optional reorder of SNAD species blocks to match species_list
        _lto = species_list if lammps_type_order is None else lammps_type_order
        if _lto != species_list:
            idx_map = [_lto.index(s) for s in species_list]
            dB_no_c0 = dB_no_c0[:, idx_map, :, :]

        dB = pad_snad_with_c0_zero(dB_no_c0)  # shape (N,S,3,K_with_c0)
        K_from_snad = dB.shape[-1]
        if K_from_snad != K_with_c0:
            raise ValueError(
                f"Feature count mismatch: B has K={K_with_c0} (after adding C0), "
                f"SNAD (padded) has K={K_from_snad}."
            )

        # forces
        F = np.asarray(entry["forces"], dtype=float)
        units = entry.get("force_units", "kcal/mol/Å")
        F = convert_forces(F, units)

        # store
        all_B.append(B)
        all_species_lists.append(atom_species)
        all_dB.append(dB)
        all_forces.append(F.reshape(-1))  # flatten xyz per atom

    K = all_B[0].shape[1]

    # Normalizer on all structures (per species)
    normalizer = SnapNormalizer.fit(all_B, all_species_lists, species_list)

    # Build design matrix: energies then forces (no intercept anywhere)
    X_energy_rows = []
    for B, atom_species in zip(all_B, all_species_lists):
        Bn = normalizer.normalize_B(B, atom_species)
        X_energy_rows.append(energy_row_from_B(Bn, atom_species, species_list))
    X_energy = np.vstack(X_energy_rows)                   # (N_struct, S*K)
    y_energy = np.array(energies, dtype=float)

    X_force_rows = []
    y_force_rows = []
    for dB, F in zip(all_dB, all_forces):
        dB_scaled = normalizer.scale_snad(dB, species_list)
        Xf = force_rows_from_snad(dB_scaled)              # (3N, S*K)
        X_force_rows.append(Xf)
        y_force_rows.append(F)
    X_forces = np.vstack(X_force_rows)                    # (sum 3N, S*K)
    y_forces = np.concatenate(y_force_rows, axis=0)

    # Apply squared weights and stack
    X = np.vstack([w_energy_sq * X_energy, w_forces_sq * X_forces])   # (N_struct + sum 3N, S*K)
    y = np.concatenate([w_energy_sq * y_energy, w_forces_sq * y_forces])

    # Fit ridge WITHOUT intercept (C0 is the only constant feature)
    model = Ridge(alpha=lambda_en, fit_intercept=False)
    model.fit(X, y)

    beta_flat = model.coef_          # (S*K,)
    beta_matrix = beta_flat.reshape(S, K)
    return beta_matrix, model, normalizer, K


# ============================================================
# Prediction & evaluation
# ============================================================
def predict_energy(
    B: np.ndarray,
    atom_species: List[str],
    species_list: List[str],
    beta: np.ndarray,
    normalizer: SnapNormalizer
) -> float:
    Bn = normalizer.normalize_B(B, atom_species)
    x = energy_row_from_B(Bn, atom_species, species_list)
    return float(x @ beta.reshape(-1))


def predict_forces(
    dB_no_c0: np.ndarray,
    species_list: List[str],
    beta: np.ndarray,
    normalizer: SnapNormalizer,
    lammps_type_order: Optional[List[str]] = None
) -> np.ndarray:
    """Predict forces (kcal/mol/Å) for one structure given raw SNAD (no C0)."""
    _lto = species_list if lammps_type_order is None else lammps_type_order
    if _lto != species_list:
        idx_map = [_lto.index(s) for s in species_list]
        dB_no_c0 = dB_no_c0[:, idx_map, :, :]
    dB = pad_snad_with_c0_zero(dB_no_c0)
    dB_scaled = normalizer.scale_snad(dB, species_list)
    Xf = force_rows_from_snad(dB_scaled)
    F = Xf @ beta.reshape(-1)
    return F.reshape(-1, 3)


def evaluate_training_set(
    training_set: List[Dict],
    species_list: List[str],
    beta: np.ndarray,
    normalizer: SnapNormalizer,
    K: int,
    lammps_type_order: Optional[List[str]] = None
) -> Dict[str, float]:
    yE_true, yE_pred = [], []
    yF_true, yF_pred = [], []

    for entry in training_set:
        B = read_sna_dump_add_c0(entry["bispec_file"])
        yE_true.append(float(entry["energy"]))
        yE_pred.append(predict_energy(B, entry["atom_species"], species_list, beta, normalizer))

        dB_no_c0 = read_snad_dump(entry["snad_file"], n_species=len(species_list))
        F_pred = predict_forces(dB_no_c0, species_list, beta, normalizer, lammps_type_order)
        F_true = convert_forces(
            np.asarray(entry["forces"], float),
            entry.get("force_units", "kcal/mol/Å")
        )
        yF_true.append(F_true.reshape(-1))
        yF_pred.append(F_pred.reshape(-1))

    yE_true = np.array(yE_true)
    yE_pred = np.array(yE_pred)
    yF_true = np.concatenate(yF_true)
    yF_pred = np.concatenate(yF_pred)

    rmse_E = float(np.sqrt(mean_squared_error(yE_true, yE_pred)))
    rmse_F = float(np.sqrt(mean_squared_error(yF_true, yF_pred)))
    return {"rmse_energy": rmse_E, "rmse_force": rmse_F}


# ============================================================
# Extra reporting (energies): per-structure & incremental RMSE (E+F)
# ============================================================
def _compute_rmse_energy(y_true, y_pred, n_atoms):
    rmse_total = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse_per_atom = np.sqrt(
        np.mean(((np.array(y_pred) - np.array(y_true)) / np.array(n_atoms)) ** 2)
    )
    return rmse_total, rmse_per_atom


def incremental_rmse_over_structures_energy_forces(
    training_set: List[Dict],
    species_list: List[str],
    lambda_en: float = 0.1,
    w_energy_sq: float = 1.0,
    w_forces_sq: float = 1.0,
    lammps_type_order: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    For k = 1..N_struct, fit on the first k structures (ENERGIES + FORCES, no intercept),
    using the same normalization pipeline as the main fit, then compute training RMSEs
    on those k structures. Returns a DataFrame with columns:
        ['k', 'rmse_energy', 'rmse_energy_per_atom', 'rmse_force']
    """
    S = len(species_list)

    # Pre-read all data to avoid repeated IO
    all_B, all_species, all_E = [], [], []
    all_dB_no_c0, all_F = [], []
    all_n_atoms = []

    for entry in training_set:
        B = read_sna_dump_add_c0(entry["bispec_file"])
        all_B.append(B)
        atom_species = entry["atom_species"]
        all_species.append(atom_species)
        all_n_atoms.append(len(atom_species))

        all_E.append(float(entry["energy"]))

        dB_nc0 = read_snad_dump(entry["snad_file"], n_species=S)
        _lto = species_list if lammps_type_order is None else lammps_type_order
        if _lto != species_list:
            idx_map = [_lto.index(s) for s in species_list]
            dB_nc0 = dB_nc0[:, idx_map, :, :]
        all_dB_no_c0.append(dB_nc0)

        F = np.asarray(entry["forces"], float)
        units = entry.get("force_units", "kcal/mol/Å")
        F = convert_forces(F, units)
        all_F.append(F.reshape(-1))

    rows = []
    for k in range(1, len(training_set) + 1):
        Bk = all_B[:k]
        Spk = all_species[:k]
        Ek = np.array(all_E[:k], dtype=float)
        dB_nc0_k = all_dB_no_c0[:k]
        Fk = all_F[:k]
        n_atoms_k = all_n_atoms[:k]

        # Normalizer on first k
        norm_k = SnapNormalizer.fit(Bk, Spk, species_list)

        # Energies design
        X_energy_rows = []
        for B, atom_species in zip(Bk, Spk):
            Bn = norm_k.normalize_B(B, atom_species)
            X_energy_rows.append(energy_row_from_B(Bn, atom_species, species_list))
        X_energy = np.vstack(X_energy_rows)
        y_energy = Ek

        # Forces design
        X_force_rows, y_force_rows = [], []
        for dB_nc0, F in zip(dB_nc0_k, Fk):
            dB = pad_snad_with_c0_zero(dB_nc0)
            dB_scaled = norm_k.scale_snad(dB, species_list)
            Xf = force_rows_from_snad(dB_scaled)
            X_force_rows.append(Xf)
            y_force_rows.append(F)
        X_forces = np.vstack(X_force_rows) if X_force_rows else np.zeros((0, X_energy.shape[1]))
        y_forces = np.concatenate(y_force_rows) if y_force_rows else np.zeros((0,))

        # Stack with squared weights
        Xk = np.vstack([w_energy_sq * X_energy, w_forces_sq * X_forces])
        yk = np.concatenate([w_energy_sq * y_energy, w_forces_sq * y_forces])

        # Fit (no intercept)
        model_k = Ridge(alpha=lambda_en, fit_intercept=False)
        model_k.fit(Xk, yk)

        # Predictions on same k
        yE_pred = X_energy @ model_k.coef_
        yF_pred = X_forces @ model_k.coef_ if y_forces.size else np.array([])

        # RMSEs
        rmse_E = float(np.sqrt(mean_squared_error(y_energy, yE_pred)))
        rmse_E_per_atom = float(
            np.sqrt(np.mean(((yE_pred - y_energy) / np.array(n_atoms_k)) ** 2))
        )
        rmse_F = float(np.sqrt(mean_squared_error(y_forces, yF_pred))) if y_forces.size else float("nan")

        rows.append(
            {
                "k": k,
                "rmse_energy": rmse_E,
                "rmse_energy_per_atom": rmse_E_per_atom,
                "rmse_force": rmse_F,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Helpers for pasting forces
# ============================================================
def _reshape_forces_auto(flat: np.ndarray, n_atoms: int) -> np.ndarray:
    """
    Reshape flat length 3N vector to (N,3). Tries both conventions and
    picks the one with smaller |sum of forces| (net force should be ~0).
    """
    if flat.size != 3 * n_atoms:
        raise ValueError(f"Expected {3*n_atoms} numbers for forces, got {flat.size}.")
    F1 = flat.reshape(n_atoms, 3)        # (fx,fy,fz) per atom
    F2 = flat.reshape(3, n_atoms).T      # all fx, then all fy, then all fz
    s1 = np.linalg.norm(F1.sum(axis=0))
    s2 = np.linalg.norm(F2.sum(axis=0))
    return F1 if s1 <= s2 else F2

# ============================================================
# Main (example with one structure; extend this pattern)
# ============================================================
if __name__ == "__main__":


    # -------- Structure 1 --------
    energy_1 = -145449.016173# kcal/mol
    _force_text_1 = """
 -0.000054024218
  0.000062194475
  0.000002189817
  0.000065502927
  0.000135077217
  0.000004962816
 -0.000005409172
 -0.000171209125
 -0.000007797793
 -0.000151917646
  0.000087496270
 -0.000001708094
  0.000116996571
 -0.000011357489
  0.000011495352
  0.000034270551
 -0.000087920153
 -0.000007893622
  0.000047772423
  0.000078911948
 -0.000003582591
 -0.000060162023
  0.000069564846
  0.000002589904
 -0.000094716051
  0.000012015828
 -0.000001272678
 -0.000023724407
 -0.000095202194
  0.000003048796
  0.000033631515
 -0.000087366561
 -0.000005308244
  0.000091779529
  0.000007794940
  0.000003276337
"""
    N_atoms_1 = 12
    atom_species_1 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces1_flat = np.fromstring(_force_text_1, sep=' ', dtype=float)
    forces1 = _reshape_forces_auto(forces1_flat, N_atoms_1)
    force_units_1 = "Eh/bohr"
    print("Net F (struct 1, kcal/mol/Å):", convert_forces(forces1, force_units_1).sum(axis=0))

    # -------- Structure 2 --------
    energy_2 = -145448.872789# kcal/mol
    _force_text_2 = """
  0.001132983914
  0.002272272588
 -0.000765304639
 -0.001375464020
  0.003273014097
  0.000588202450
  0.000371486974
 -0.000164056899
 -0.000234881240
  0.000465086995
  0.001104721470
  0.000095064457
  0.002378192544
 -0.001784418751
 -0.000459396719
  0.001536805832
 -0.001400804088
  0.000578659401
 -0.001213087571
 -0.001115641174
  0.000164119745
  0.001261920792
 -0.002651879403
 -0.000118353258
 -0.000423388599
 -0.000150067079
  0.000035706551
 -0.000824609487
 -0.001386199141
  0.000065099232
 -0.001755050177
  0.002102642899
  0.000084304699
 -0.001554877186
 -0.000099584520
 -0.000033220674
"""
    N_atoms_2 = 12
    atom_species_2 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces2_flat = np.fromstring(_force_text_2, sep=' ', dtype=float)
    forces2 = _reshape_forces_auto(forces2_flat, N_atoms_2)
    force_units_2 = "Eh/bohr"
    print("Net F (struct 2, kcal/mol/Å):", convert_forces(forces2, force_units_2).sum(axis=0))

    # -------- Structure 3 --------
    energy_3 = -145448.867985# kcal/mol
    _force_text_3 = """
  0.001820128323
 -0.002151376982
 -0.000119424618
  0.001069076030
 -0.001402668753
 -0.000373292788
  0.002914597817
  0.001467471254
  0.000313098497
  0.000436294177
 -0.000306994300
  0.000154715362
  0.002272845289
 -0.001774410017
  0.000133504280
  0.000216146788
  0.002188535716
 -0.000349128339
 -0.000279591365
  0.000105953643
  0.000136971504
 -0.000881195959
  0.000875709372
  0.000136929644
 -0.003821140252
 -0.000506156179
 -0.000051687059
 -0.000399560209
 -0.000182632133
 -0.000158749168
 -0.001559172069
  0.001826646399
 -0.000003641112
 -0.001788428547
 -0.000140078029
  0.000180703797
"""
    N_atoms_3 = 12
    atom_species_3 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces3_flat = np.fromstring(_force_text_3, sep=' ', dtype=float)
    forces3 = _reshape_forces_auto(forces3_flat, N_atoms_3)
    force_units_3 = "Eh/bohr"
    print("Net F (struct 3, kcal/mol/Å):", convert_forces(forces3, force_units_3).sum(axis=0))

    # -------- Structure 4 --------
    energy_4 = -145448.831701# kcal/mol
    _force_text_4 = """
 -0.000064887729
  0.000021878715
 -0.000053049473
 -0.002353493437
  0.003537601991
 -0.000122521361
  0.000928418518
  0.001540197766
  0.000309084889
  0.002728951425
  0.000999807561
 -0.000244482325
 -0.000479565815
  0.002839734181
 -0.000050175143
  0.004434248753
 -0.001200463919
 -0.000060977245
  0.000345934400
  0.000893519862
  0.000030089765
  0.002572130742
 -0.004187311719
  0.000064831141
 -0.002590342478
 -0.000015430932
 -0.000114866912
 -0.001863547988
 -0.002348406769
  0.000124887167
  0.001363618865
 -0.002098858372
  0.000032294907
 -0.005021465246
  0.000017731637
  0.000084884586
"""
    N_atoms_4 = 12
    atom_species_4 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces4_flat = np.fromstring(_force_text_4, sep=' ', dtype=float)
    forces4 = _reshape_forces_auto(forces4_flat, N_atoms_4)
    force_units_4 = "Eh/bohr"
    print("Net F (struct 4, kcal/mol/Å):", convert_forces(forces4, force_units_4).sum(axis=0))

    # -------- Structure 5 --------
    energy_5 = -145448.812901# kcal/mol
    _force_text_5 = """
 -0.001775265086
  0.000138981563
 -0.000250284363
  0.006503904499
 -0.005742763815
 -0.000110766769
  0.000784321036
  0.000984010925
  0.000472467082
  0.003491300235
 -0.001279417863
  0.000925300268
  0.000342077519
  0.000515652547
 -0.000190583088
 -0.002625232401
 -0.000161729517
  0.000308729288
 -0.000592253815
 -0.000654766565
  0.000214932204
 -0.003695649484
  0.006970020830
 -0.000262207866
 -0.002027467709
  0.000225463110
 -0.000166441718
 -0.001680420268
 -0.001403726956
 -0.000626592599
 -0.000264760348
  0.000330729630
 -0.000046317073
  0.001539445810
  0.000077546119
 -0.000268235366
"""
    N_atoms_5 = 12
    atom_species_5 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces5_flat = np.fromstring(_force_text_5, sep=' ', dtype=float)
    forces5 = _reshape_forces_auto(forces5_flat, N_atoms_5)
    force_units_5 = "Eh/bohr"
    print("Net F (struct 5, kcal/mol/Å):", convert_forces(forces5, force_units_5).sum(axis=0))

    # -------- Structure 6 --------
    energy_6 = -145448.793801# kcal/mol
    _force_text_6 = """
 -0.002173228005
  0.002973522591
 -0.000118991111
 -0.000057961615
  0.003923494893
  0.001231938710
 -0.007574030658
  0.001004233210
 -0.000269719310
  0.002791407406
 -0.003617800554
 -0.000213688284
 -0.003907324490
  0.006572649186
  0.000281296512
 -0.000078650590
 -0.001732608276
 -0.000238519696
 -0.000052107666
 -0.002278131825
 -0.000209797409
  0.001398863514
 -0.003004246948
 -0.000487062103
  0.006208186756
  0.000338177991
 -0.000058689475
  0.000815267765
  0.000757210847
  0.000083701490
  0.002562569171
 -0.003845051806
 -0.000170368825
  0.000067008417
 -0.001091449309
  0.000169899502
"""
    N_atoms_6 = 12
    atom_species_6 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces6_flat = np.fromstring(_force_text_6, sep=' ', dtype=float)
    forces6 = _reshape_forces_auto(forces6_flat, N_atoms_6)
    force_units_6 = "Eh/bohr"
    print("Net F (struct 6, kcal/mol/Å):", convert_forces(forces6, force_units_6).sum(axis=0))

    # -------- Structure 7 --------
    energy_7 = -145448.686830# kcal/mol
    _force_text_7 = """
 -0.006009102014
 -0.004598117478
  0.000032449087
  0.003392054972
 -0.007953316284
  0.000532267068
  0.002005536534
 -0.000812381832
 -0.000508393821
 -0.005671849788
 -0.002066320518
  0.000227842796
 -0.003904585470
  0.008617401402
  0.000010476800
 -0.002250528421
 -0.003659616303
 -0.000442549805
  0.003525730832
  0.004943227533
 -0.000449364623
 -0.002106558833
  0.004042874225
  0.000164271520
 -0.000124146776
 -0.000184415936
 -0.000163729652
  0.004595769200
  0.006893775079
  0.000329241298
  0.004123805614
 -0.005545184845
 -0.000418735326
  0.002423874144
  0.000322074962
  0.000686224663
"""
    N_atoms_7 = 12
    atom_species_7 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces7_flat = np.fromstring(_force_text_7, sep=' ', dtype=float)
    forces7 = _reshape_forces_auto(forces7_flat, N_atoms_7)
    force_units_7 = "Eh/bohr"
    print("Net F (struct 7, kcal/mol/Å):", convert_forces(forces7, force_units_7).sum(axis=0))

    # -------- Structure 8 --------
    energy_8 = -145448.606682# kcal/mol
    _force_text_8 = """
 -0.003896832858
 -0.003869995736
  0.002212208549
  0.003811231020
 -0.000442350559
  0.001013795095
 -0.002798408146
 -0.004347749264
 -0.001068925569
 -0.006889987842
  0.004894823839
  0.000771746370
  0.002369965718
  0.003133249581
  0.000059878465
  0.004262066495
 -0.006916428003
 -0.001224419668
  0.001803599035
  0.006246150803
 -0.001749768381
 -0.002190262172
 -0.001683862458
  0.000288692432
  0.005036643553
 -0.000546202090
 -0.000542708206
 -0.000210254149
  0.001374899848
 -0.000139138761
 -0.000765717062
  0.000930646919
 -0.000067282827
 -0.000532043585
  0.001226817125
  0.000445922505
"""
    N_atoms_8 = 12
    atom_species_8 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces8_flat = np.fromstring(_force_text_8, sep=' ', dtype=float)
    forces8 = _reshape_forces_auto(forces8_flat, N_atoms_8)
    force_units_8 = "Eh/bohr"
    print("Net F (struct 8, kcal/mol/Å):", convert_forces(forces8, force_units_8).sum(axis=0))

    # -------- Structure 9 --------
    energy_9 = -145448.411171# kcal/mol
    _force_text_9 = """
  0.006676494177
 -0.007791913587
  0.000754738835
 -0.003905162402
  0.000768873507
 -0.001641300256
 -0.010821740132
  0.002218446699
 -0.001079780885
  0.001194153383
 -0.015004342971
  0.000371174102
 -0.000301569420
  0.003714747370
  0.004807062149
 -0.011018808827
  0.001272987635
 -0.001711765092
  0.003217251821
  0.005723221615
  0.000842286552
 -0.003261234487
  0.004381590642
  0.000196952029
  0.007844111722
  0.000488917331
  0.000829116312
  0.003993821687
  0.008297992617
 -0.001183286015
  0.001087542435
 -0.004336130284
 -0.001150866166
  0.005295140031
  0.000265609427
 -0.001034331571
"""
    N_atoms_9 = 12
    atom_species_9 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces9_flat = np.fromstring(_force_text_9, sep=' ', dtype=float)
    forces9 = _reshape_forces_auto(forces9_flat, N_atoms_9)
    force_units_9 = "Eh/bohr"
    print("Net F (struct 9, kcal/mol/Å):", convert_forces(forces9, force_units_9).sum(axis=0))

    # -------- Structure 10 --------
    energy_10 = -145448.510522# kcal/mol
    _force_text_10 = """
 -0.005738464311
 -0.000258195328
 -0.000514712462
  0.009925605354
 -0.002899912500
 -0.001239550211
 -0.005567610587
 -0.009251964690
  0.000592665010
  0.005846717179
 -0.001217743294
  0.000699459508
 -0.005021642282
  0.000013698615
 -0.001309255203
 -0.005772574176
  0.003127213155
 -0.001959404106
 -0.000247629668
 -0.001194371214
  0.001060842116
 -0.000015078239
  0.002574845610
  0.000392597740
  0.001854330949
  0.001153247270
  0.000792810203
 -0.000674397822
  0.007462950062
 -0.001091310668
  0.000166829634
 -0.000038153387
  0.001861801211
  0.005243913945
  0.000528385702
  0.000714056857
"""
    N_atoms_10 = 12
    atom_species_10 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces10_flat = np.fromstring(_force_text_10, sep=' ', dtype=float)
    forces10 = _reshape_forces_auto(forces10_flat, N_atoms_10)
    force_units_10 = "Eh/bohr"
    print("Net F (struct 10, kcal/mol/Å):", convert_forces(forces10, force_units_10).sum(axis=0))

    # -------- Structure 11 --------
    energy_11 = -145448.368100# kcal/mol
    _force_text_11 = """
  0.006442060626
 -0.006464348442
 -0.001236605582
 -0.007786021136
  0.001610029071
  0.001374787645
  0.008374446196
 -0.002209006363
  0.001964001252
 -0.000005530955
 -0.001102783589
 -0.005245014691
 -0.002325690723
  0.002638425003
  0.007896574737
 -0.008172054662
  0.008836962099
 -0.004488369087
 -0.003594129447
 -0.004726663021
  0.001120463651
  0.003328203465
 -0.002153703746
 -0.000762757626
 -0.000091436008
  0.001547485461
 -0.000195089113
 -0.003210468558
  0.001573288534
  0.000585802595
  0.000474610392
 -0.001489813643
 -0.001671172310
  0.006566010798
  0.001940128631
  0.000657378528
"""
    N_atoms_11 = 12
    atom_species_11 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces11_flat = np.fromstring(_force_text_11, sep=' ', dtype=float)
    forces11 = _reshape_forces_auto(forces11_flat, N_atoms_11)
    force_units_11 = "Eh/bohr"
    print("Net F (struct 11, kcal/mol/Å):", convert_forces(forces11, force_units_11).sum(axis=0))

    # -------- Structure 12 --------
    energy_12 = -145448.300366# kcal/mol
    _force_text_12 = """
  0.001830066637
  0.002735463836
 -0.004498591584
 -0.001290614797
  0.001184007157
 -0.001389304816
 -0.001772991325
 -0.002090414298
 -0.002588958077
  0.001181936177
  0.001324457864
 -0.002560146770
  0.001970954732
 -0.009152191440
 -0.002320282040
  0.001901569439
 -0.000144002242
  0.001443301901
  0.001408332746
  0.002360566590
  0.002653935169
 -0.000207575608
 -0.002476956928
  0.002709127871
 -0.000899749909
  0.000316584979
  0.001633210125
  0.001370885061
  0.002056306804
  0.002168182320
 -0.003348302254
  0.004147429747
  0.002792713700
 -0.002144510887
 -0.000261252069
 -0.000043187804
"""
    N_atoms_12 = 12
    atom_species_12 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces12_flat = np.fromstring(_force_text_12, sep=' ', dtype=float)
    forces12 = _reshape_forces_auto(forces12_flat, N_atoms_12)
    force_units_12 = "Eh/bohr"
    print("Net F (struct 12, kcal/mol/Å):", convert_forces(forces12, force_units_12).sum(axis=0))

    # -------- Structure 13 --------
    energy_13 = -145447.887935# kcal/mol
    _force_text_13 = """
 -0.002907827666
 -0.005302741420
  0.006739971612
 -0.005495289718
 -0.005040342747
 -0.007109089441
  0.011456160889
 -0.005354792003
  0.003516258584
 -0.002273053619
  0.001818436268
  0.001009019055
 -0.005078899592
  0.005270940044
  0.003632613093
 -0.001476891626
 -0.007969900684
 -0.007504247081
 -0.001136899757
  0.002887050436
 -0.002536795407
  0.002748203137
  0.003366829575
  0.001944066977
  0.000018294146
  0.003530935696
 -0.000945971965
  0.002548051846
  0.001925853843
 -0.000118788415
  0.001158874944
  0.003616536969
 -0.003488404043
  0.000439277022
  0.001251194025
  0.004861367031
"""
    N_atoms_13 = 12
    atom_species_13 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces13_flat = np.fromstring(_force_text_13, sep=' ', dtype=float)
    forces13 = _reshape_forces_auto(forces13_flat, N_atoms_13)
    force_units_13 = "Eh/bohr"
    print("Net F (struct 13, kcal/mol/Å):", convert_forces(forces13, force_units_13).sum(axis=0))

    # -------- Structure 14 --------
    energy_14 = -145448.434448# kcal/mol
    _force_text_14 = """
 -0.009427611999
  0.007039256365
  0.002768034607
  0.004439661293
 -0.001414591109
  0.000124786682
 -0.001361461394
 -0.003461997124
 -0.000111605786
 -0.002595509488
  0.002100038992
  0.000314499758
  0.007088200972
 -0.000781707545
  0.002763492994
  0.005584620255
 -0.003604378934
 -0.002492430595
 -0.001760107256
 -0.003393532189
 -0.000933412684
  0.002968249506
 -0.000749925120
 -0.000855446034
 -0.000247473572
  0.001360208027
 -0.000443218011
 -0.000922245014
  0.000053030057
  0.000200504748
 -0.005077318123
  0.000192431322
 -0.001995749430
  0.001310994819
  0.002661167268
  0.000660543749
"""
    N_atoms_14 = 12
    atom_species_14 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces14_flat = np.fromstring(_force_text_14, sep=' ', dtype=float)
    forces14 = _reshape_forces_auto(forces14_flat, N_atoms_14)
    force_units_14 = "Eh/bohr"
    print("Net F (struct 14, kcal/mol/Å):", convert_forces(forces14, force_units_14).sum(axis=0))

    # -------- Structure 15 --------
    energy_15 = -145447.885708# kcal/mol
    _force_text_15 = """
 -0.006041428871
 -0.008882488424
 -0.000803699341
  0.001550745913
  0.004388723643
 -0.002002946677
 -0.003743704196
  0.002251936845
  0.008143031258
 -0.016639623096
 -0.010946673352
 -0.005063334009
  0.003340964980
  0.004768056632
  0.001953254660
  0.010803450994
 -0.005547229724
  0.003355309434
  0.008017163920
  0.011821814476
 -0.000157442399
  0.001041456511
 -0.005243404206
 -0.000457929029
 -0.003616162911
 -0.000384649810
 -0.002856880336
  0.003167811323
  0.005711068265
  0.001326633467
  0.004949515244
  0.001393788859
 -0.001570073122
 -0.002830189805
  0.000669056803
 -0.001865923907
"""
    N_atoms_15 = 12
    atom_species_15 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces15_flat = np.fromstring(_force_text_15, sep=' ', dtype=float)
    forces15 = _reshape_forces_auto(forces15_flat, N_atoms_15)
    force_units_15 = "Eh/bohr"
    print("Net F (struct 15, kcal/mol/Å):", convert_forces(forces15, force_units_15).sum(axis=0))

    # -------- Structure 16 --------
    energy_16 = -145447.296199# kcal/mol
    _force_text_16 = """
 -0.012938087037
  0.012733476304
  0.004474531156
  0.002424631410
  0.016289912980
 -0.005026566992
  0.005449419411
  0.009613198785
 -0.001901479679
 -0.002469289172
 -0.017055996131
  0.005737075976
 -0.004938680170
 -0.012584343385
 -0.005099611709
  0.008896178573
 -0.006719998164
  0.000651758599
 -0.006525794738
  0.001593446894
 -0.000343693218
  0.002063882144
 -0.006911578889
 -0.000347191898
 -0.009250540110
  0.001801432906
  0.001893544160
  0.005597714273
  0.007396484860
 -0.000824937803
  0.004295307727
 -0.004872261377
 -0.000039873121
  0.007395257681
 -0.001283774804
  0.000826444529
"""
    N_atoms_16 = 12
    atom_species_16 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces16_flat = np.fromstring(_force_text_16, sep=' ', dtype=float)
    forces16 = _reshape_forces_auto(forces16_flat, N_atoms_16)
    force_units_16 = "Eh/bohr"
    print("Net F (struct 16, kcal/mol/Å):", convert_forces(forces16, force_units_16).sum(axis=0))

    # -------- Structure 17 --------
    energy_17 = -145447.180285# kcal/mol
    _force_text_17 = """
  0.011057729581
  0.006126104870
  0.003626336899
 -0.013111680905
  0.006696908188
 -0.005491329251
 -0.008906641747
  0.003318846762
 -0.001442374032
  0.002148359686
 -0.009979596846
 -0.000731214861
 -0.002693563977
 -0.010886880404
 -0.004104887167
 -0.009579463286
 -0.000225442082
 -0.002861507841
  0.000147644166
 -0.006374549454
  0.001026028996
  0.002260131128
 -0.000995492317
  0.002720823963
  0.015575409283
 -0.001787323034
  0.001750508767
  0.003506670891
 -0.002956148564
  0.001342000408
 -0.010132165534
  0.016438229139
  0.004374072045
  0.009727570710
  0.000625343747
 -0.000208457931
"""
    N_atoms_17 = 12
    atom_species_17 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces17_flat = np.fromstring(_force_text_17, sep=' ', dtype=float)
    forces17 = _reshape_forces_auto(forces17_flat, N_atoms_17)
    force_units_17 = "Eh/bohr"
    print("Net F (struct 17, kcal/mol/Å):", convert_forces(forces17, force_units_17).sum(axis=0))

    # -------- Structure 18 --------
    energy_18 = -145446.855557# kcal/mol
    _force_text_18 = """
  0.017310208563
 -0.004532649420
  0.001722095903
  0.001815312593
  0.011245621551
 -0.000051550912
  0.007195328405
 -0.011046208496
  0.002120193752
  0.027284757989
  0.005544900831
 -0.003114434225
 -0.010237161147
  0.015869664632
 -0.001483726044
  0.000741806067
 -0.009236931036
 -0.004798950015
  0.000834914799
  0.000854081116
  0.001182782283
 -0.004176146415
  0.001118987300
 -0.000479660450
 -0.013739317785
 -0.000040190947
 -0.000026751852
 -0.010546711765
 -0.018465491780
  0.000932066567
 -0.007704444758
  0.006895285504
  0.003142692717
 -0.008778546541
  0.001792930757
  0.000855242268
"""
    N_atoms_18 = 12
    atom_species_18 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces18_flat = np.fromstring(_force_text_18, sep=' ', dtype=float)
    forces18 = _reshape_forces_auto(forces18_flat, N_atoms_18)
    force_units_18 = "Eh/bohr"
    print("Net F (struct 18, kcal/mol/Å):", convert_forces(forces18, force_units_18).sum(axis=0))

    # -------- Structure 19 --------
    energy_19 = -145447.516967# kcal/mol
    _force_text_19 = """
 -0.007595840956
  0.001593167718
  0.004541200822
  0.011724507869
 -0.015710742569
 -0.000473548473
 -0.011421671742
  0.002599240168
 -0.000825148429
  0.007855766949
  0.007244276805
  0.000820889938
 -0.004148321708
  0.012487666789
  0.000130129500
 -0.026617483098
  0.000514039092
 -0.003102902538
  0.005810036658
  0.000314646827
 -0.001169990225
 -0.003454897552
  0.001157727402
 -0.001147517919
  0.010240386574
  0.002089499260
  0.000600558100
 -0.006988307198
 -0.002623453351
  0.000100411865
  0.008528497353
 -0.012729253533
 -0.000516580975
  0.016067326847
  0.003063185385
  0.001042498334
"""
    N_atoms_19 = 12
    atom_species_19 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces19_flat = np.fromstring(_force_text_19, sep=' ', dtype=float)
    forces19 = _reshape_forces_auto(forces19_flat, N_atoms_19)
    force_units_19 = "Eh/bohr"
    print("Net F (struct 19, kcal/mol/Å):", convert_forces(forces19, force_units_19).sum(axis=0))

    # -------- Structure 20 --------
    energy_20 = -145445.748672# kcal/mol
    _force_text_20 = """
  0.021513118442
  0.006046520486
 -0.007173923103
  0.017954835634
  0.026692728003
 -0.001043221246
  0.000214854983
 -0.020554072690
 -0.008120079645
  0.027626794476
  0.002449554839
  0.006756503026
 -0.011061585401
  0.025055995860
  0.001433270515
 -0.031339421410
  0.003362477721
  0.002829627578
 -0.011323785015
 -0.009513713719
  0.005609957615
  0.001297356354
 -0.011304171476
 -0.000339195144
 -0.023258372801
  0.002931915107
  0.005689763705
 -0.005001885067
 -0.003837633116
 -0.004018098396
  0.006243197728
 -0.017411003762
  0.000552832191
  0.007134892063
 -0.003918597255
 -0.002177437091
"""
    N_atoms_20 = 12
    atom_species_20 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces20_flat = np.fromstring(_force_text_20, sep=' ', dtype=float)
    forces20 = _reshape_forces_auto(forces20_flat, N_atoms_20)
    force_units_20 = "Eh/bohr"
    print("Net F (struct 20, kcal/mol/Å):", convert_forces(forces20, force_units_20).sum(axis=0))

    # -------- Structure 21 --------
    energy_21 = -145446.960680# kcal/mol
    _force_text_21 = """
 -0.004405533928
  0.002149135833
 -0.004379539130
  0.000011118712
  0.005684323964
  0.003745520378
 -0.015870952968
 -0.002552392561
  0.003316417311
 -0.005349908474
 -0.015865802252
 -0.003615132976
  0.029175395842
  0.005087100878
  0.001720320221
 -0.012954618477
 -0.002616765556
 -0.005333827905
 -0.000310919379
  0.003495842417
  0.001268062489
  0.002326150243
 -0.009415129986
 -0.000686198961
  0.013789196665
  0.009768622546
 -0.000384032376
 -0.002023005710
  0.002301480122
 -0.000564032436
 -0.012628135995
  0.004996395250
  0.002171854492
  0.008241213483
 -0.003032810656
  0.002740588891
"""
    N_atoms_21 = 12
    atom_species_21 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces21_flat = np.fromstring(_force_text_21, sep=' ', dtype=float)
    forces21 = _reshape_forces_auto(forces21_flat, N_atoms_21)
    force_units_21 = "Eh/bohr"
    print("Net F (struct 21, kcal/mol/Å):", convert_forces(forces21, force_units_21).sum(axis=0))

    # -------- Structure 22 --------
    energy_22 = -145446.461249# kcal/mol
    _force_text_22 = """
 -0.015148462097
 -0.016765561937
 -0.000035366225
  0.004303928280
  0.017242723531
  0.001413724455
 -0.030712746475
 -0.001271427663
 -0.003705728644
  0.003936216421
 -0.009380770291
 -0.007824793048
 -0.004573027881
 -0.008931553628
  0.010281262580
  0.020511483956
 -0.002433580921
 -0.001829347281
  0.008873080891
  0.011535815166
 -0.001347643967
  0.005894919525
 -0.013751969596
  0.003962142032
  0.008900600150
 -0.003159253942
  0.000983807680
  0.004465276116
  0.014641983799
  0.000800292609
 -0.002925692846
  0.008257792475
 -0.001181717631
 -0.003525576037
  0.004015803012
 -0.001516632565
"""
    N_atoms_22 = 12
    atom_species_22 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces22_flat = np.fromstring(_force_text_22, sep=' ', dtype=float)
    forces22 = _reshape_forces_auto(forces22_flat, N_atoms_22)
    force_units_22 = "Eh/bohr"
    print("Net F (struct 22, kcal/mol/Å):", convert_forces(forces22, force_units_22).sum(axis=0))

    # -------- Structure 23 --------
    energy_23 = -145445.662370# kcal/mol
    _force_text_23 = """
  0.015579021460
  0.013501828678
 -0.001330360813
  0.016846663166
 -0.003181046542
  0.003515972111
 -0.002056861355
  0.009837579103
 -0.006262985685
 -0.023954278645
 -0.014026430269
  0.009183951824
  0.004927176382
  0.007247967273
 -0.021726762841
  0.000236445124
 -0.004624058158
  0.009621500606
 -0.014107241630
 -0.010483040417
 -0.005071505965
  0.001650093441
  0.004845126340
 -0.005724743841
  0.000715190800
 -0.005002648915
 -0.006992655761
  0.002642241229
  0.002235135162
  0.009249537484
 -0.000233366849
 -0.001554616301
  0.007934845785
 -0.002245083111
  0.001204204051
  0.007603207095
"""
    N_atoms_23 = 12
    atom_species_23 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces23_flat = np.fromstring(_force_text_23, sep=' ', dtype=float)
    forces23 = _reshape_forces_auto(forces23_flat, N_atoms_23)
    force_units_23 = "Eh/bohr"
    print("Net F (struct 23, kcal/mol/Å):", convert_forces(forces23, force_units_23).sum(axis=0))

    # -------- Structure 24 --------
    energy_24 = -145445.065319# kcal/mol
    _force_text_24 = """
 -0.011884906893
  0.010748677797
 -0.003033565982
  0.003764970045
 -0.021468998599
 -0.008972290086
  0.020155071109
 -0.010739966157
 -0.000653997543
 -0.029279641972
  0.020020882756
  0.003618030409
  0.023615068553
 -0.033056020812
 -0.017537624094
  0.015150842871
  0.018443757948
  0.010096931935
  0.006266383219
  0.011545394894
  0.007902712115
 -0.004775951475
  0.015087428441
  0.004361415535
 -0.006299408259
  0.002495442840
  0.002278449375
 -0.005293386682
 -0.017352801463
 -0.004211683449
 -0.005733134969
  0.006197938501
  0.008337440990
 -0.005685905537
 -0.001921736143
 -0.002185819211
"""
    N_atoms_24 = 12
    atom_species_24 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces24_flat = np.fromstring(_force_text_24, sep=' ', dtype=float)
    forces24 = _reshape_forces_auto(forces24_flat, N_atoms_24)
    force_units_24 = "Eh/bohr"
    print("Net F (struct 24, kcal/mol/Å):", convert_forces(forces24, force_units_24).sum(axis=0))

    # -------- Structure 25 --------
    energy_25 = -145445.932525# kcal/mol
    _force_text_25 = """
 -0.011260416155
  0.033188898980
  0.000156997819
 -0.023768356642
 -0.026811166478
 -0.007414132083
  0.010025451777
  0.030317663466
  0.002261287821
  0.004225068710
  0.002301977313
  0.001758049870
  0.004123890212
 -0.005163180911
  0.004487472412
  0.003098596389
 -0.018377325674
  0.002948977894
  0.002123145564
 -0.006429364527
  0.000695261119
  0.005961865261
  0.004762713492
  0.001250738792
  0.000727327084
 -0.000423310775
  0.002181192340
  0.003008319935
 -0.008639690225
 -0.002555418189
 -0.005326215862
 -0.006803065996
 -0.003306186205
  0.007061323720
  0.002075851341
 -0.002464241591
"""
    N_atoms_25 = 12
    atom_species_25 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces25_flat = np.fromstring(_force_text_25, sep=' ', dtype=float)
    forces25 = _reshape_forces_auto(forces25_flat, N_atoms_25)
    force_units_25 = "Eh/bohr"
    print("Net F (struct 25, kcal/mol/Å):", convert_forces(forces25, force_units_25).sum(axis=0))

    # -------- Structure 26 --------
    energy_26 = -145445.194218# kcal/mol
    _force_text_26 = """
 -0.014212074459
  0.005535562183
 -0.009972614288
  0.006170746650
 -0.004639432603
 -0.011490789087
  0.020128931362
 -0.013013176630
  0.032963345354
  0.004367288479
 -0.011956166756
  0.043348305346
 -0.013967782620
  0.017244055504
 -0.040433003760
  0.005515930566
 -0.003812670545
  0.015362370059
  0.005663789998
 -0.000258674084
 -0.012218933390
  0.002246057631
 -0.000201003778
  0.002045479209
 -0.011757449417
  0.008397706844
 -0.009874729831
  0.003081355249
 -0.002027116175
 -0.006724456179
  0.000416041431
 -0.002348040074
  0.007180456914
 -0.007652834871
  0.007078956107
 -0.010185430344
"""
    N_atoms_26 = 12
    atom_species_26 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces26_flat = np.fromstring(_force_text_26, sep=' ', dtype=float)
    forces26 = _reshape_forces_auto(forces26_flat, N_atoms_26)
    force_units_26 = "Eh/bohr"
    print("Net F (struct 26, kcal/mol/Å):", convert_forces(forces26, force_units_26).sum(axis=0))

    # -------- Structure 27 --------
    energy_27 = -145445.424084# kcal/mol
    _force_text_27 = """
 -0.014561475901
  0.030343692666
  0.005145164587
 -0.013222792051
 -0.018158103297
  0.008436399678
  0.003207279006
  0.022977427296
 -0.000970768772
 -0.006457113720
 -0.001197099074
 -0.001266621405
 -0.005655423631
 -0.001667896976
  0.004022762415
  0.015774268021
 -0.017793962043
 -0.000016347157
  0.001085322201
 -0.010235541992
  0.001195110689
 -0.000519099624
  0.020599214346
 -0.001921410417
  0.011296344814
 -0.011171774535
 -0.010678780472
  0.002622119058
 -0.001433724264
  0.001476713736
  0.004458571793
 -0.010888804113
 -0.001806125918
  0.001972000034
 -0.001373427991
 -0.003616096969
"""
    N_atoms_27 = 12
    atom_species_27 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces27_flat = np.fromstring(_force_text_27, sep=' ', dtype=float)
    forces27 = _reshape_forces_auto(forces27_flat, N_atoms_27)
    force_units_27 = "Eh/bohr"
    print("Net F (struct 27, kcal/mol/Å):", convert_forces(forces27, force_units_27).sum(axis=0))

    # -------- Structure 28 --------
    energy_28 = -145444.977933# kcal/mol
    _force_text_28 = """
 -0.008442344033
  0.002100074064
 -0.000787291066
  0.005922363519
  0.012792290636
 -0.041871173293
 -0.015652069510
 -0.006519581558
  0.005880988901
  0.025917287278
 -0.000340646173
 -0.005022830694
 -0.017534093084
  0.005007591774
 -0.000717675445
  0.005857025398
 -0.014056593735
  0.021618326269
  0.005166834076
 -0.001782463625
  0.004654942093
  0.012060757434
 -0.001684835048
  0.007299715762
 -0.002382811074
  0.002224729544
 -0.000892259935
 -0.002329056252
 -0.002483773974
  0.001146791564
 -0.003965299938
  0.005258020937
  0.001248876078
 -0.004618593821
 -0.000514812840
  0.007441589764
"""
    N_atoms_28 = 12
    atom_species_28 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces28_flat = np.fromstring(_force_text_28, sep=' ', dtype=float)
    forces28 = _reshape_forces_auto(forces28_flat, N_atoms_28)
    force_units_28 = "Eh/bohr"
    print("Net F (struct 28, kcal/mol/Å):", convert_forces(forces28, force_units_28).sum(axis=0))

    # -------- Structure 29 --------
    energy_29 = -145444.699923# kcal/mol
    _force_text_29 = """
 -0.012916472024
 -0.001002871071
  0.003720612383
 -0.013942795186
 -0.000840460823
  0.014774071754
  0.005967864173
  0.013638171181
  0.014762219361
 -0.001086797179
  0.009838991492
  0.012624724687
 -0.013504953697
  0.003580246611
  0.002625522498
 -0.024579027416
 -0.028338513274
 -0.007997030004
  0.016336341023
  0.011146942091
  0.000158771286
  0.018122139756
 -0.005929443862
 -0.027591532865
 -0.003786105804
 -0.008784075374
 -0.007239978737
  0.008147925373
  0.002115745615
 -0.001030387210
  0.010982546267
 -0.004339266547
 -0.010117916923
  0.010259334721
  0.008914533967
  0.005310923760
"""
    N_atoms_29 = 12
    atom_species_29 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces29_flat = np.fromstring(_force_text_29, sep=' ', dtype=float)
    forces29 = _reshape_forces_auto(forces29_flat, N_atoms_29)
    force_units_29 = "Eh/bohr"
    print("Net F (struct 29, kcal/mol/Å):", convert_forces(forces29, force_units_29).sum(axis=0))

    # -------- Structure 30 --------
    energy_30 = -145443.953162# kcal/mol
    _force_text_30 = """
 -0.021367390284
  0.000761798171
 -0.030923174925
  0.011888114225
  0.028447925295
  0.009781001446
 -0.003036243926
 -0.015360295395
  0.001937964427
  0.002406753271
 -0.018560817739
  0.006204550638
  0.023112206478
  0.015047387170
  0.006201081979
 -0.004844841339
  0.005741534769
  0.016469991039
  0.007156468344
 -0.012714907976
  0.011604052420
 -0.011804738799
 -0.015794511417
 -0.006526208875
  0.014099656896
  0.006519497482
  0.007789255677
 -0.003148943844
  0.009291567865
 -0.001768465265
 -0.015218642954
 -0.009516994789
 -0.010560955539
  0.000757601922
  0.006137816538
 -0.010209093018
"""
    N_atoms_30 = 12
    atom_species_30 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces30_flat = np.fromstring(_force_text_30, sep=' ', dtype=float)
    forces30 = _reshape_forces_auto(forces30_flat, N_atoms_30)
    force_units_30 = "Eh/bohr"
    print("Net F (struct 30, kcal/mol/Å):", convert_forces(forces30, force_units_30).sum(axis=0))

    # -------- Structure 31 --------
    energy_31 = -145442.970579# kcal/mol
    _force_text_31 = """
 -0.027452614984
  0.003664649730
  0.017580718442
  0.006556525885
 -0.017516061211
  0.016256150874
 -0.010308282114
  0.033717881761
 -0.039794871232
 -0.024849362572
 -0.017877545469
  0.013944512585
  0.017278050065
 -0.010181358817
  0.011699187212
 -0.005361849811
  0.009056578185
 -0.005893127580
  0.001913414945
  0.002099682648
 -0.012072775079
  0.028478504270
  0.005462966754
 -0.012662013516
  0.002418964396
 -0.016362233128
  0.020344910267
  0.013364103453
  0.002774605466
  0.000698360235
 -0.008017674559
  0.004043010905
 -0.005243535273
  0.005980221031
  0.001117823166
 -0.004857516933
"""
    N_atoms_31 = 12
    atom_species_31 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces31_flat = np.fromstring(_force_text_31, sep=' ', dtype=float)
    forces31 = _reshape_forces_auto(forces31_flat, N_atoms_31)
    force_units_31 = "Eh/bohr"
    print("Net F (struct 31, kcal/mol/Å):", convert_forces(forces31, force_units_31).sum(axis=0))

    # -------- Structure 32 --------
    energy_32 = -145441.111928# kcal/mol
    _force_text_32 = """
  0.009151770753
 -0.023308198817
  0.014778420562
 -0.005181922866
 -0.004671763781
 -0.008769273110
  0.000608157977
  0.031664723242
  0.007611340742
  0.004780358486
 -0.023234984786
  0.001551876012
  0.022053958834
 -0.002186451803
 -0.016167957968
 -0.027315683364
  0.029646745613
 -0.007098147908
  0.008746518902
  0.012799583018
 -0.014132376951
  0.001977846140
  0.004990919159
  0.000446534734
 -0.010741214079
 -0.009234906207
 -0.000774613463
 -0.001036966109
 -0.011303039526
  0.012626130758
  0.005267993121
  0.004392870574
 -0.004613445323
 -0.008310817804
 -0.009555496688
  0.014541511916
"""
    N_atoms_32 = 12
    atom_species_32 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces32_flat = np.fromstring(_force_text_32, sep=' ', dtype=float)
    forces32 = _reshape_forces_auto(forces32_flat, N_atoms_32)
    force_units_32 = "Eh/bohr"
    print("Net F (struct 32, kcal/mol/Å):", convert_forces(forces32, force_units_32).sum(axis=0))

    # -------- Structure 33 --------
    energy_33 = -145443.097733# kcal/mol
    _force_text_33 = """
 -0.017892670114
  0.016356989317
 -0.018423354286
  0.007505269194
 -0.031435500815
  0.025952086873
  0.020222747604
  0.008796689164
 -0.002750222117
 -0.033940623679
 -0.002338459550
 -0.004055290385
 -0.018359832853
 -0.020769563494
  0.033222374470
  0.007345340253
  0.000781735135
 -0.007179947792
  0.007270340221
  0.001966111279
  0.001229031687
  0.007250682252
  0.002412961804
 -0.001828444941
 -0.014148179460
  0.004803069720
 -0.006078735492
  0.021300710998
 -0.004999421988
  0.004168281267
  0.015999873580
  0.010032320593
 -0.011933356964
 -0.002553657979
  0.014393068844
 -0.012322422315
"""
    N_atoms_33 = 12
    atom_species_33 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces33_flat = np.fromstring(_force_text_33, sep=' ', dtype=float)
    forces33 = _reshape_forces_auto(forces33_flat, N_atoms_33)
    force_units_33 = "Eh/bohr"
    print("Net F (struct 33, kcal/mol/Å):", convert_forces(forces33, force_units_33).sum(axis=0))

    # -------- Structure 34 --------
    energy_34 = -145444.069412# kcal/mol
    _force_text_34 = """
 -0.009282531435
  0.027567597972
  0.011937089346
  0.009633013817
  0.004492376300
 -0.017136707122
 -0.011738793940
 -0.007796345081
  0.020083446492
  0.008987437254
 -0.015621796352
 -0.037196365797
  0.004757112821
 -0.006045554693
  0.003159423557
  0.014266138535
 -0.004905744536
 -0.017649135542
  0.006875167054
 -0.006739259101
  0.006362961105
 -0.005482900074
 -0.005692188884
  0.010803363343
 -0.012220299423
  0.002543146947
 -0.002669117069
  0.003471355935
  0.008659161141
  0.016873617165
 -0.000558597835
  0.005346338694
 -0.000981558855
 -0.008707102709
 -0.001807732399
  0.006412983390
"""
    N_atoms_34 = 12
    atom_species_34 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces34_flat = np.fromstring(_force_text_34, sep=' ', dtype=float)
    forces34 = _reshape_forces_auto(forces34_flat, N_atoms_34)
    force_units_34 = "Eh/bohr"
    print("Net F (struct 34, kcal/mol/Å):", convert_forces(forces34, force_units_34).sum(axis=0))

    # -------- Structure 35 --------
    energy_35 = -145441.230618# kcal/mol
    _force_text_35 = """
 -0.003782487216
  0.016687878685
 -0.003868351373
  0.012367222620
 -0.037392769250
 -0.016742353741
 -0.000413478256
 -0.011756405299
  0.048786225608
  0.009363840887
 -0.026741765194
 -0.027659714509
  0.015661566827
 -0.043176743289
  0.025072160942
 -0.001795036999
  0.010548123209
 -0.015078619482
 -0.004130596936
  0.013763864057
 -0.005068316024
 -0.003767608476
  0.012724391571
  0.004386652596
 -0.004825007780
  0.013148298845
 -0.001555384939
 -0.006100297835
  0.026140206141
 -0.016059003387
 -0.007768700889
  0.019374555823
  0.001575342608
 -0.004809415943
  0.006680364709
  0.006211361710
"""
    N_atoms_35 = 12
    atom_species_35 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces35_flat = np.fromstring(_force_text_35, sep=' ', dtype=float)
    forces35 = _reshape_forces_auto(forces35_flat, N_atoms_35)
    force_units_35 = "Eh/bohr"
    print("Net F (struct 35, kcal/mol/Å):", convert_forces(forces35, force_units_35).sum(axis=0))

    # -------- Structure 36 --------
    energy_36 = -145441.592152# kcal/mol
    _force_text_36 = """
  0.024129337601
  0.002147165644
  0.026130422152
 -0.002003040606
 -0.003945667565
 -0.019535586577
 -0.026017229906
 -0.032406510744
  0.016051221379
  0.012529657396
  0.003602192560
 -0.013048369115
  0.008749959076
  0.036075148333
 -0.032023371774
 -0.013613486680
 -0.007882280429
 -0.007114459450
 -0.003548594529
  0.005694359952
  0.004452451203
 -0.002817750127
  0.000636760584
  0.007680267245
 -0.000571323266
 -0.002683942808
 -0.002328246491
 -0.003741406460
 -0.002737759122
  0.013386161343
 -0.001337167235
 -0.000845453947
 -0.004279626161
  0.008241044741
  0.002345987548
  0.010629136246
"""
    N_atoms_36 = 12
    atom_species_36 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces36_flat = np.fromstring(_force_text_36, sep=' ', dtype=float)
    forces36 = _reshape_forces_auto(forces36_flat, N_atoms_36)
    force_units_36 = "Eh/bohr"
    print("Net F (struct 36, kcal/mol/Å):", convert_forces(forces36, force_units_36).sum(axis=0))

    # -------- Structure 37 --------
    energy_37 = -145440.365565# kcal/mol
    _force_text_37 = """
  0.030837930855
  0.051158249508
 -0.025269612787
  0.009865021240
  0.002799072373
  0.001680970232
  0.032111812937
 -0.022073554628
 -0.002914968602
  0.007581075736
 -0.008664344771
 -0.007942596549
 -0.070323248522
  0.005821388598
  0.014567234306
  0.006227100786
 -0.042386632547
  0.007102477824
 -0.002938401076
 -0.003171760246
  0.002666809529
 -0.018388845504
 -0.009877794519
  0.009882673260
 -0.003269297839
  0.003814421979
 -0.002479788969
 -0.004269839456
  0.001214093271
  0.005906709830
  0.017721641773
  0.003528917347
 -0.002384191707
 -0.005154950920
  0.017837943637
 -0.000815716366
"""
    N_atoms_37 = 12
    atom_species_37 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces37_flat = np.fromstring(_force_text_37, sep=' ', dtype=float)
    forces37 = _reshape_forces_auto(forces37_flat, N_atoms_37)
    force_units_37 = "Eh/bohr"
    print("Net F (struct 37, kcal/mol/Å):", convert_forces(forces37, force_units_37).sum(axis=0))

    # -------- Structure 38 --------
    energy_38 = -145439.911796# kcal/mol
    _force_text_38 = """
  0.019552008744
  0.035247076637
  0.021919619234
 -0.013383373465
  0.013183693523
 -0.001752088151
  0.001601496533
 -0.015429526978
 -0.034826088439
 -0.002637765786
 -0.014861664726
  0.028582208574
 -0.033540898273
  0.053319228580
 -0.029662772442
  0.023285967671
 -0.044497111616
 -0.000864872145
 -0.015246848514
 -0.010108509842
 -0.002719222515
 -0.004281254228
  0.005080695669
 -0.006490629410
  0.004680624465
  0.003706386279
  0.006310907540
 -0.000359477203
  0.004905949620
  0.002272767973
  0.020876599774
 -0.032376498431
  0.018596911023
 -0.000547079713
  0.001830281270
 -0.001366741239
"""
    N_atoms_38 = 12
    atom_species_38 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces38_flat = np.fromstring(_force_text_38, sep=' ', dtype=float)
    forces38 = _reshape_forces_auto(forces38_flat, N_atoms_38)
    force_units_38 = "Eh/bohr"
    print("Net F (struct 38, kcal/mol/Å):", convert_forces(forces38, force_units_38).sum(axis=0))

    # -------- Structure 39 --------
    energy_39 = -145440.718843# kcal/mol
    _force_text_39 = """
  0.007101098753
 -0.011750803987
  0.000465996316
  0.012687230537
 -0.007776125243
  0.013362720711
 -0.000246256580
 -0.003586059703
 -0.010890463078
  0.012632682017
 -0.001693987573
 -0.000950745155
  0.001383095576
 -0.021911307306
 -0.010761405542
 -0.019858513422
  0.010313270194
  0.017737177429
 -0.022224601230
  0.022773098991
 -0.010206061854
  0.012206638859
 -0.003095709679
 -0.001549612706
 -0.014696706788
  0.002643570156
  0.000751498701
  0.008063064592
  0.008210795151
  0.004727058701
 -0.010247270523
  0.012746437529
 -0.001687381321
  0.013199538226
 -0.006873178515
 -0.000998782197
"""
    N_atoms_39 = 12
    atom_species_39 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces39_flat = np.fromstring(_force_text_39, sep=' ', dtype=float)
    forces39 = _reshape_forces_auto(forces39_flat, N_atoms_39)
    force_units_39 = "Eh/bohr"
    print("Net F (struct 39, kcal/mol/Å):", convert_forces(forces39, force_units_39).sum(axis=0))

    # -------- Structure 40 --------
    energy_40 = -145437.374991# kcal/mol
    _force_text_40 = """
 -0.034356676872
  0.000597356736
  0.008816693976
 -0.010676985974
 -0.035903968141
 -0.006730555118
  0.029713443356
  0.027879497789
  0.014782231383
 -0.049951952071
 -0.009758140070
 -0.006022465285
  0.016516465564
 -0.020728322807
  0.001520953751
  0.010882980166
  0.025020985393
 -0.004082862999
 -0.011601061983
  0.006014492432
  0.005602999912
 -0.007406151645
  0.014394204412
 -0.001872838236
 -0.002016720890
 -0.015401719354
  0.000552401106
  0.015299697139
  0.006698129093
 -0.004139572738
  0.041719972455
  0.002478762339
 -0.002716861029
  0.001876990778
 -0.001291277826
 -0.005710124728
"""
    N_atoms_40 = 12
    atom_species_40 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces40_flat = np.fromstring(_force_text_40, sep=' ', dtype=float)
    forces40 = _reshape_forces_auto(forces40_flat, N_atoms_40)
    force_units_40 = "Eh/bohr"
    print("Net F (struct 40, kcal/mol/Å):", convert_forces(forces40, force_units_40).sum(axis=0))

    # -------- Structure 41 --------
    energy_41 = -145439.928272# kcal/mol
    _force_text_41 = """
 -0.020305749726
 -0.019670246723
  0.045699300667
  0.005310622002
 -0.013267451837
 -0.026726050095
  0.011623202306
 -0.016289401712
  0.011071707893
  0.021106191859
 -0.006622897405
 -0.039388039478
 -0.012548848120
  0.000624355915
  0.012749187687
  0.007886822685
  0.019925554299
 -0.023810530559
  0.006841367489
  0.010217425126
 -0.008792406237
  0.002784479137
  0.006706536274
 -0.004746494702
 -0.008972348938
  0.014283924275
  0.008105261475
 -0.016787808936
 -0.000776476071
  0.024267377182
  0.009241443616
  0.006798148657
 -0.004063822925
 -0.006179373371
 -0.001929470822
  0.005634509092
"""
    N_atoms_41 = 12
    atom_species_41 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces41_flat = np.fromstring(_force_text_41, sep=' ', dtype=float)
    forces41 = _reshape_forces_auto(forces41_flat, N_atoms_41)
    force_units_41 = "Eh/bohr"
    print("Net F (struct 41, kcal/mol/Å):", convert_forces(forces41, force_units_41).sum(axis=0))

    # -------- Structure 42 --------
    energy_42 = -145438.622725# kcal/mol
    _force_text_42 = """
 -0.042708681374
 -0.016414420932
 -0.006848606802
  0.013657056935
 -0.019227827868
 -0.027652201227
  0.016397026017
  0.003106607769
  0.006402886501
  0.012095953408
 -0.031153438662
 -0.013620758207
  0.012622716168
  0.006723278447
  0.010642726088
  0.002392444498
 -0.023140970516
 -0.009203858629
  0.030204097069
  0.014466498185
  0.009776336638
  0.002140889108
  0.014128121412
  0.014009165779
 -0.012869254640
  0.018961700080
  0.013171331341
 -0.003474519388
  0.005450728680
 -0.007468368777
 -0.006444836089
 -0.003580054406
 -0.006148892878
 -0.024012891713
  0.030679777791
  0.016940240167
"""
    N_atoms_42 = 12
    atom_species_42 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces42_flat = np.fromstring(_force_text_42, sep=' ', dtype=float)
    forces42 = _reshape_forces_auto(forces42_flat, N_atoms_42)
    force_units_42 = "Eh/bohr"
    print("Net F (struct 42, kcal/mol/Å):", convert_forces(forces42, force_units_42).sum(axis=0))

    # -------- Structure 43 --------
    energy_43 = -145439.486109# kcal/mol
    _force_text_43 = """
 -0.012880311469
 -0.000924983827
  0.008990964474
 -0.023374304931
  0.011481203393
  0.030195577199
  0.020797436648
 -0.038634745909
 -0.008758976916
  0.001783216753
  0.014648322563
 -0.007623961227
  0.002565536495
 -0.020205288495
 -0.007263845685
 -0.012060908192
  0.007157416899
 -0.017823746132
  0.015191197518
  0.006452839086
  0.005964132885
  0.007116278952
  0.011338393828
 -0.004733824459
  0.018461588390
  0.000600415375
 -0.000839669980
 -0.009710711482
  0.006169515679
  0.004265316677
 -0.005316937174
  0.004193586427
 -0.000388058069
 -0.002572081519
 -0.002276675041
 -0.001983908777
"""
    N_atoms_43 = 12
    atom_species_43 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces43_flat = np.fromstring(_force_text_43, sep=' ', dtype=float)
    forces43 = _reshape_forces_auto(forces43_flat, N_atoms_43)
    force_units_43 = "Eh/bohr"
    print("Net F (struct 43, kcal/mol/Å):", convert_forces(forces43, force_units_43).sum(axis=0))

    # -------- Structure 44 --------
    energy_44 = -145434.925555# kcal/mol
    _force_text_44 = """
  0.039500350352
 -0.033871908730
 -0.048878029955
  0.004875091167
  0.005017261492
 -0.001186299051
  0.030485364550
 -0.029975218662
 -0.024622602758
  0.030814762798
  0.014741275322
  0.006393531943
 -0.043210991565
  0.009903484990
  0.029272063921
  0.042005736590
  0.003186708579
 -0.005986162840
  0.003450608454
  0.011083197418
  0.009117374450
 -0.019518316754
  0.007124295891
  0.011712340560
 -0.017021446719
 -0.007081243329
 -0.008977204653
 -0.000716275371
  0.022881997364
  0.023686847512
 -0.017452322419
  0.008253785053
  0.010053054592
 -0.053212561062
 -0.011263635407
 -0.000584913729
"""
    N_atoms_44 = 12
    atom_species_44 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces44_flat = np.fromstring(_force_text_44, sep=' ', dtype=float)
    forces44 = _reshape_forces_auto(forces44_flat, N_atoms_44)
    force_units_44 = "Eh/bohr"
    print("Net F (struct 44, kcal/mol/Å):", convert_forces(forces44, force_units_44).sum(axis=0))

    # -------- Structure 45 --------
    energy_45 = -145440.865586# kcal/mol
    _force_text_45 = """
 -0.003128665003
  0.006594735560
  0.006551655937
  0.012271334421
 -0.000653509290
 -0.007411546544
  0.028117181464
 -0.004860470338
  0.030047240584
 -0.042278635265
  0.002602143849
 -0.020382346406
  0.025734388533
 -0.013053922467
  0.004368284424
 -0.041547895537
  0.005320543134
 -0.022696662896
  0.005490570349
 -0.005389571695
  0.000457824161
 -0.008703743909
  0.007664732459
 -0.005286769098
 -0.007766057046
 -0.010368303968
 -0.002145107776
  0.003764246524
 -0.007597113469
 -0.000462559523
  0.003512594688
  0.020274733049
  0.001330062011
  0.024534680784
 -0.000533996824
  0.015629925129
"""
    N_atoms_45 = 12
    atom_species_45 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces45_flat = np.fromstring(_force_text_45, sep=' ', dtype=float)
    forces45 = _reshape_forces_auto(forces45_flat, N_atoms_45)
    force_units_45 = "Eh/bohr"
    print("Net F (struct 45, kcal/mol/Å):", convert_forces(forces45, force_units_45).sum(axis=0))

    # -------- Structure 46 --------
    energy_46 = -145436.341650# kcal/mol
    _force_text_46 = """
  0.015768862264
  0.015272797075
 -0.051272249519
 -0.006583293236
  0.015511730833
  0.001632615806
  0.012545279136
  0.014480532389
 -0.014176532646
  0.014421426798
  0.014365764057
  0.034080706059
 -0.014430230888
 -0.023408029018
 -0.066560818334
 -0.009183424087
 -0.010290684566
  0.029512129160
 -0.003618566414
 -0.002644681087
  0.008830167057
  0.000954634016
 -0.020082621032
  0.051632174100
 -0.010539158442
 -0.017152702274
 -0.005721531304
 -0.000210691465
  0.010766853091
  0.002074167008
 -0.001926129382
 -0.004564061752
  0.008664682894
  0.002801291705
  0.007745102287
  0.001304489729
"""
    N_atoms_46 = 12
    atom_species_46 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces46_flat = np.fromstring(_force_text_46, sep=' ', dtype=float)
    forces46 = _reshape_forces_auto(forces46_flat, N_atoms_46)
    force_units_46 = "Eh/bohr"
    print("Net F (struct 46, kcal/mol/Å):", convert_forces(forces46, force_units_46).sum(axis=0))

    # -------- Structure 47 --------
    energy_47 = -145438.663480# kcal/mol
    _force_text_47 = """
 -0.001114497335
 -0.017231673151
 -0.015690767931
 -0.008837765453
  0.040960742516
 -0.003716601317
 -0.011377011694
  0.001727912836
 -0.008884528518
 -0.019432231258
  0.002149096820
 -0.010980790232
 -0.009532940581
 -0.014121687136
 -0.002020155008
  0.014693242533
  0.025575372244
  0.005061025946
 -0.006157770112
  0.000395767638
  0.003269299060
  0.011086423920
 -0.024785243178
  0.012960768729
  0.014603504162
 -0.005228176443
  0.006358669018
 -0.000273926812
 -0.002635537466
  0.006609581958
  0.000105813382
 -0.022698195968
 -0.000402607080
  0.016237159265
  0.015891621282
  0.007436105364
"""
    N_atoms_47 = 12
    atom_species_47 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces47_flat = np.fromstring(_force_text_47, sep=' ', dtype=float)
    forces47 = _reshape_forces_auto(forces47_flat, N_atoms_47)
    force_units_47 = "Eh/bohr"
    print("Net F (struct 47, kcal/mol/Å):", convert_forces(forces47, force_units_47).sum(axis=0))

    # -------- Structure 48 --------
    energy_48 = -145436.032128# kcal/mol
    _force_text_48 = """
  0.001223122885
  0.026520844083
  0.017120185794
 -0.009945171765
 -0.030651796282
 -0.003029439890
  0.010329767464
 -0.019667711458
  0.007685179085
 -0.009128626935
 -0.003281757482
 -0.001125580853
  0.025144952399
  0.000136494192
  0.008379596231
 -0.008489963326
 -0.029056957349
 -0.027956688332
  0.001413019384
 -0.019009000450
 -0.002127221836
  0.005507917938
  0.015980346808
  0.004162834401
  0.007938650575
  0.020005587186
 -0.008639801466
 -0.015004370025
  0.022742325921
 -0.001343623446
 -0.012808681251
 -0.025896983436
  0.002691929846
  0.003819382659
  0.042178608284
  0.004182630466
"""
    N_atoms_48 = 12
    atom_species_48 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces48_flat = np.fromstring(_force_text_48, sep=' ', dtype=float)
    forces48 = _reshape_forces_auto(forces48_flat, N_atoms_48)
    force_units_48 = "Eh/bohr"
    print("Net F (struct 48, kcal/mol/Å):", convert_forces(forces48, force_units_48).sum(axis=0))

    # -------- Structure 49 --------
    energy_49 = -145434.653476# kcal/mol
    _force_text_49 = """
  0.003748115453
 -0.005603700857
  0.012573655738
 -0.000267509770
  0.003195002432
  0.002608893453
 -0.007535686001
  0.012406863138
 -0.039755397628
 -0.064230869816
  0.040529864817
  0.123651955477
  0.015290737799
  0.000304690899
  0.038331810428
  0.010854033659
 -0.025511115162
 -0.007172397555
  0.001774761931
  0.000613710018
 -0.020868962802
  0.003617197682
 -0.000581647844
 -0.010649985505
  0.010770852138
 -0.008368519600
  0.003742431069
  0.040997014649
 -0.048197227654
 -0.082060908982
 -0.000249442050
  0.003595193819
 -0.015085293341
 -0.014769205657
  0.027616885975
 -0.005315800378
"""
    N_atoms_49 = 12
    atom_species_49 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces49_flat = np.fromstring(_force_text_49, sep=' ', dtype=float)
    forces49 = _reshape_forces_auto(forces49_flat, N_atoms_49)
    force_units_49 = "Eh/bohr"
    print("Net F (struct 49, kcal/mol/Å):", convert_forces(forces49, force_units_49).sum(axis=0))

    # -------- Structure 50 --------
    energy_50 = -145435.159113# kcal/mol
    _force_text_50 = """
 -0.001115467358
 -0.026954518075
 -0.000364386876
 -0.030807908506
  0.028086333775
  0.003621256713
  0.023246965231
 -0.019276415201
  0.001596942922
 -0.033299952394
  0.032106097354
  0.001582957629
 -0.000382127693
  0.038335865984
 -0.019235826466
 -0.010746670692
 -0.032327539772
  0.023966912480
 -0.003971411379
  0.017260608601
 -0.001484352930
  0.061122869563
 -0.031653925589
 -0.002877575428
 -0.003407628523
  0.005476687314
 -0.000281126993
 -0.003910538080
 -0.010641245042
 -0.000692580969
  0.002934517496
 -0.003255500323
  0.003636029445
  0.000337352365
  0.002843550987
 -0.009468249529
"""
    N_atoms_50 = 12
    atom_species_50 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces50_flat = np.fromstring(_force_text_50, sep=' ', dtype=float)
    forces50 = _reshape_forces_auto(forces50_flat, N_atoms_50)
    force_units_50 = "Eh/bohr"
    print("Net F (struct 50, kcal/mol/Å):", convert_forces(forces50, force_units_50).sum(axis=0))

    # -------- Structure 51 --------
    energy_51 = -145435.791489# kcal/mol
    _force_text_51 = """
  0.029138932637
  0.027257980063
  0.018779335861
 -0.028240058343
 -0.027777550225
  0.003462128276
  0.001443940281
  0.013023288684
 -0.011710237618
  0.023114421766
 -0.008545027015
  0.007776483751
  0.001009943637
 -0.002633125082
  0.005851270605
  0.023170211450
  0.001219319722
 -0.014050397223
 -0.011098695799
 -0.011877798097
 -0.010099178456
  0.012870940883
  0.005316017295
 -0.000252410390
  0.003167446611
  0.008933946695
  0.004583177877
 -0.031186881569
 -0.016599846598
 -0.005364279859
  0.012841602711
  0.010957455898
  0.000586092625
 -0.036231804249
  0.000725338705
  0.000438014546
"""
    N_atoms_51 = 12
    atom_species_51 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces51_flat = np.fromstring(_force_text_51, sep=' ', dtype=float)
    forces51 = _reshape_forces_auto(forces51_flat, N_atoms_51)
    force_units_51 = "Eh/bohr"
    print("Net F (struct 51, kcal/mol/Å):", convert_forces(forces51, force_units_51).sum(axis=0))

    # -------- Structure 52 --------
    energy_52 = -145432.908604# kcal/mol
    _force_text_52 = """
 -0.019301904503
  0.018080709369
  0.005547055977
  0.094209145957
 -0.030740331969
 -0.003882926156
 -0.066598326181
  0.011811483605
  0.007286322363
  0.003343930003
 -0.004981744051
 -0.012003257000
 -0.011624635364
 -0.004338173120
 -0.005769584836
  0.020585674183
 -0.002713255043
  0.001428867906
  0.007898772720
 -0.000891737180
  0.004576907252
 -0.016412139022
 -0.004935321054
  0.003266087571
 -0.014007000956
  0.035792203262
 -0.007784881240
  0.020292004835
 -0.015256453273
  0.013685152462
 -0.009168262715
 -0.015403981728
 -0.003808772480
 -0.009217258931
  0.013576601181
 -0.002540971814
"""
    N_atoms_52 = 12
    atom_species_52 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces52_flat = np.fromstring(_force_text_52, sep=' ', dtype=float)
    forces52 = _reshape_forces_auto(forces52_flat, N_atoms_52)
    force_units_52 = "Eh/bohr"
    print("Net F (struct 52, kcal/mol/Å):", convert_forces(forces52, force_units_52).sum(axis=0))

    # -------- Structure 53 --------
    energy_53 = -145431.741458# kcal/mol
    _force_text_53 = """
 -0.005194596656
  0.008850599061
 -0.002965201241
  0.087318965458
 -0.025889585972
 -0.018089499344
 -0.028156541682
 -0.029453809079
 -0.048301862077
 -0.029946711474
  0.003797797346
  0.025410771544
 -0.011295313952
  0.001387872883
  0.017460208645
 -0.005537716123
 -0.018858191658
  0.014505318987
  0.004273154917
  0.003755855484
  0.009204069586
 -0.055723683966
  0.020658626109
  0.006391018649
  0.027717997078
  0.016238138109
  0.010962078036
 -0.004486245797
 -0.000426191771
  0.006149351673
  0.014056736861
  0.004564182435
 -0.016870124278
  0.006973955333
  0.015374707038
 -0.003856130173
"""
    N_atoms_53 = 12
    atom_species_53 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces53_flat = np.fromstring(_force_text_53, sep=' ', dtype=float)
    forces53 = _reshape_forces_auto(forces53_flat, N_atoms_53)
    force_units_53 = "Eh/bohr"
    print("Net F (struct 53, kcal/mol/Å):", convert_forces(forces53, force_units_53).sum(axis=0))

    # -------- Structure 54 --------
    energy_54 = -145431.875518# kcal/mol
    _force_text_54 = """
  0.023929433924
 -0.024375656619
  0.015604375951
 -0.030002734965
 -0.034762651483
 -0.001151694455
  0.027812393042
  0.037065468818
  0.011430383001
  0.007570605586
  0.010538177475
 -0.028157721381
  0.013324556163
  0.087003038785
 -0.000377610835
 -0.010392073434
 -0.075100099764
  0.008448398322
  0.002884954141
  0.011526596091
 -0.006168588031
  0.009756501392
 -0.002677273989
  0.000335756258
 -0.024123079107
 -0.006463774950
 -0.003125843519
 -0.012784617650
 -0.004694273037
  0.009067007459
  0.019966449843
 -0.007944121535
  0.003616024175
 -0.027942388929
  0.009884570192
 -0.009520486948
"""
    N_atoms_54 = 12
    atom_species_54 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces54_flat = np.fromstring(_force_text_54, sep=' ', dtype=float)
    forces54 = _reshape_forces_auto(forces54_flat, N_atoms_54)
    force_units_54 = "Eh/bohr"
    print("Net F (struct 54, kcal/mol/Å):", convert_forces(forces54, force_units_54).sum(axis=0))

    # -------- Structure 55 --------
    energy_55 = -145436.229735# kcal/mol
    _force_text_55 = """
 -0.005258501060
  0.005681317265
  0.023307679530
  0.005820979647
 -0.038589942173
  0.012176211625
 -0.020936195593
  0.015100046703
  0.032936733703
  0.018626383919
  0.022468800846
 -0.026983203144
  0.015654580357
  0.015676504166
 -0.023055999017
 -0.009064595576
 -0.001417975215
 -0.008682164111
 -0.002139626961
  0.007246849344
 -0.004324165049
 -0.002856292282
  0.002377362356
 -0.005614719648
  0.007872278583
 -0.012944294798
 -0.006549398278
 -0.010554845304
  0.000250532690
  0.007624984055
  0.004188992947
  0.006719457413
 -0.018282115314
 -0.001353158674
 -0.022568658609
  0.017446155646
"""
    N_atoms_55 = 12
    atom_species_55 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces55_flat = np.fromstring(_force_text_55, sep=' ', dtype=float)
    forces55 = _reshape_forces_auto(forces55_flat, N_atoms_55)
    force_units_55 = "Eh/bohr"
    print("Net F (struct 55, kcal/mol/Å):", convert_forces(forces55, force_units_55).sum(axis=0))

    # -------- Structure 56 --------
    energy_56 = -145432.811831# kcal/mol
    _force_text_56 = """
 -0.026496524994
  0.004220661170
  0.028260796037
  0.047593310642
 -0.005054405229
 -0.001169870714
 -0.011123910911
  0.003531757682
 -0.033597615383
  0.030451177925
 -0.014410109542
 -0.023797198274
 -0.003146365491
  0.007009835874
  0.010697912801
 -0.022376178493
  0.005062226653
  0.024640600214
 -0.003616912944
 -0.002644620915
  0.003171584694
 -0.005199652031
  0.000736007124
 -0.004992901342
  0.018925490127
  0.004474752997
 -0.022735922897
 -0.004079511868
  0.004459728353
 -0.007642869983
 -0.039571563033
 -0.003426088413
  0.030563525964
  0.018640641069
 -0.003959745761
 -0.003398041128
"""
    N_atoms_56 = 12
    atom_species_56 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces56_flat = np.fromstring(_force_text_56, sep=' ', dtype=float)
    forces56 = _reshape_forces_auto(forces56_flat, N_atoms_56)
    force_units_56 = "Eh/bohr"
    print("Net F (struct 56, kcal/mol/Å):", convert_forces(forces56, force_units_56).sum(axis=0))

    # -------- Structure 57 --------
    energy_57 = -145433.439454# kcal/mol
    _force_text_57 = """
 -0.009054099139
 -0.004309365761
  0.026612488764
  0.000641184690
  0.032712519573
  0.006961932832
 -0.002687392076
  0.021372417679
 -0.043574833324
  0.009792313343
 -0.027579264004
  0.027065159454
 -0.017179009390
 -0.007348240179
  0.022011648878
  0.004066447389
 -0.005188320981
 -0.015600703850
 -0.001733385847
 -0.006576976397
  0.000736985448
  0.010221269264
 -0.029721317199
 -0.005511573955
 -0.001925211443
 -0.009251435396
  0.004532187524
  0.005636743984
  0.032633471783
 -0.018828901707
 -0.008952060217
 -0.007805622798
  0.015593519376
  0.011173199441
  0.011062133683
 -0.019997909440
"""
    N_atoms_57 = 12
    atom_species_57 = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    forces57_flat = np.fromstring(_force_text_57, sep=' ', dtype=float)
    forces57 = _reshape_forces_auto(forces57_flat, N_atoms_57)
    force_units_57 = "Eh/bohr"
    print("Net F (struct 57, kcal/mol/Å):", convert_forces(forces57, force_units_57).sum(axis=0))

    training_set = [
        {
            "bispec_file": "dump_bispec1",
            "snad_file":   "dump_derbispec1",
            "atom_species": atom_species_1,
            "energy": energy_1,
            "forces": forces1,
            "force_units": force_units_1,
        },
        {
            "bispec_file": "dump_bispec2",
            "snad_file":   "dump_derbispec2",
            "atom_species": atom_species_2,
            "energy": energy_2,
            "forces": forces2,
            "force_units": force_units_2,
        },
        {
            "bispec_file": "dump_bispec3",
            "snad_file":   "dump_derbispec3",
            "atom_species": atom_species_3,
            "energy": energy_3,
            "forces": forces3,
            "force_units": force_units_3,
        },
        {
            "bispec_file": "dump_bispec4",
            "snad_file":   "dump_derbispec4",
            "atom_species": atom_species_4,
            "energy": energy_4,
            "forces": forces4,
            "force_units": force_units_4,
        },
        {
            "bispec_file": "dump_bispec5",
            "snad_file":   "dump_derbispec5",
            "atom_species": atom_species_5,
            "energy": energy_5,
            "forces": forces5,
            "force_units": force_units_5,
        },
        {
            "bispec_file": "dump_bispec6",
            "snad_file":   "dump_derbispec6",
            "atom_species": atom_species_6,
            "energy": energy_6,
            "forces": forces6,
            "force_units": force_units_6,
        },
        {
            "bispec_file": "dump_bispec7",
            "snad_file":   "dump_derbispec7",
            "atom_species": atom_species_7,
            "energy": energy_7,
            "forces": forces7,
            "force_units": force_units_7,
        },
        {
            "bispec_file": "dump_bispec8",
            "snad_file":   "dump_derbispec8",
            "atom_species": atom_species_8,
            "energy": energy_8,
            "forces": forces8,
            "force_units": force_units_8,
        },
        {
            "bispec_file": "dump_bispec9",
            "snad_file":   "dump_derbispec9",
            "atom_species": atom_species_9,
            "energy": energy_9,
            "forces": forces9,
            "force_units": force_units_9,
        },
        {
            "bispec_file": "dump_bispec10",
            "snad_file":   "dump_derbispec10",
            "atom_species": atom_species_10,
            "energy": energy_10,
            "forces": forces10,
            "force_units": force_units_10,
        },
        {
            "bispec_file": "dump_bispec11",
            "snad_file":   "dump_derbispec11",
            "atom_species": atom_species_11,
            "energy": energy_11,
            "forces": forces11,
            "force_units": force_units_11,
        },
        {
            "bispec_file": "dump_bispec12",
            "snad_file":   "dump_derbispec12",
            "atom_species": atom_species_12,
            "energy": energy_12,
            "forces": forces12,
            "force_units": force_units_12,
        },
        {
            "bispec_file": "dump_bispec13",
            "snad_file":   "dump_derbispec13",
            "atom_species": atom_species_13,
            "energy": energy_13,
            "forces": forces13,
            "force_units": force_units_13,
        },
        {
            "bispec_file": "dump_bispec14",
            "snad_file":   "dump_derbispec14",
            "atom_species": atom_species_14,
            "energy": energy_14,
            "forces": forces14,
            "force_units": force_units_14,
        },
        {
            "bispec_file": "dump_bispec15",
            "snad_file":   "dump_derbispec15",
            "atom_species": atom_species_15,
            "energy": energy_15,
            "forces": forces15,
            "force_units": force_units_15,
        },
        {
            "bispec_file": "dump_bispec16",
            "snad_file":   "dump_derbispec16",
            "atom_species": atom_species_16,
            "energy": energy_16,
            "forces": forces16,
            "force_units": force_units_16,
        },
        {
            "bispec_file": "dump_bispec17",
            "snad_file":   "dump_derbispec17",
            "atom_species": atom_species_17,
            "energy": energy_17,
            "forces": forces17,
            "force_units": force_units_17,
        },
        {
            "bispec_file": "dump_bispec18",
            "snad_file":   "dump_derbispec18",
            "atom_species": atom_species_18,
            "energy": energy_18,
            "forces": forces18,
            "force_units": force_units_18,
        },
        {
            "bispec_file": "dump_bispec19",
            "snad_file":   "dump_derbispec19",
            "atom_species": atom_species_19,
            "energy": energy_19,
            "forces": forces19,
            "force_units": force_units_19,
        },
        {
            "bispec_file": "dump_bispec20",
            "snad_file":   "dump_derbispec20",
            "atom_species": atom_species_20,
            "energy": energy_20,
            "forces": forces20,
            "force_units": force_units_20,
        },
        {
            "bispec_file": "dump_bispec21",
            "snad_file":   "dump_derbispec21",
            "atom_species": atom_species_21,
            "energy": energy_21,
            "forces": forces21,
            "force_units": force_units_21,
        },
        {
            "bispec_file": "dump_bispec22",
            "snad_file":   "dump_derbispec22",
            "atom_species": atom_species_22,
            "energy": energy_22,
            "forces": forces22,
            "force_units": force_units_22,
        },
        {
            "bispec_file": "dump_bispec23",
            "snad_file":   "dump_derbispec23",
            "atom_species": atom_species_23,
            "energy": energy_23,
            "forces": forces23,
            "force_units": force_units_23,
        },
        {
            "bispec_file": "dump_bispec24",
            "snad_file":   "dump_derbispec24",
            "atom_species": atom_species_24,
            "energy": energy_24,
            "forces": forces24,
            "force_units": force_units_24,
        },
        {
            "bispec_file": "dump_bispec25",
            "snad_file":   "dump_derbispec25",
            "atom_species": atom_species_25,
            "energy": energy_25,
            "forces": forces25,
            "force_units": force_units_25,
        },
        {
            "bispec_file": "dump_bispec26",
            "snad_file":   "dump_derbispec26",
            "atom_species": atom_species_26,
            "energy": energy_26,
            "forces": forces26,
            "force_units": force_units_26,
        },
        {
            "bispec_file": "dump_bispec27",
            "snad_file":   "dump_derbispec27",
            "atom_species": atom_species_27,
            "energy": energy_27,
            "forces": forces27,
            "force_units": force_units_27,
        },
        {
            "bispec_file": "dump_bispec28",
            "snad_file":   "dump_derbispec28",
            "atom_species": atom_species_28,
            "energy": energy_28,
            "forces": forces28,
            "force_units": force_units_28,
        },
        {
            "bispec_file": "dump_bispec29",
            "snad_file":   "dump_derbispec29",
            "atom_species": atom_species_29,
            "energy": energy_29,
            "forces": forces29,
            "force_units": force_units_29,
        },
        {
            "bispec_file": "dump_bispec30",
            "snad_file":   "dump_derbispec30",
            "atom_species": atom_species_30,
            "energy": energy_30,
            "forces": forces30,
            "force_units": force_units_30,
        },
        {
            "bispec_file": "dump_bispec31",
            "snad_file":   "dump_derbispec31",
            "atom_species": atom_species_31,
            "energy": energy_31,
            "forces": forces31,
            "force_units": force_units_31,
        },
        {
            "bispec_file": "dump_bispec32",
            "snad_file":   "dump_derbispec32",
            "atom_species": atom_species_32,
            "energy": energy_32,
            "forces": forces32,
            "force_units": force_units_32,
        },
        {
            "bispec_file": "dump_bispec33",
            "snad_file":   "dump_derbispec33",
            "atom_species": atom_species_33,
            "energy": energy_33,
            "forces": forces33,
            "force_units": force_units_33,
        },
        {
            "bispec_file": "dump_bispec34",
            "snad_file":   "dump_derbispec34",
            "atom_species": atom_species_34,
            "energy": energy_34,
            "forces": forces34,
            "force_units": force_units_34,
        },
        {
            "bispec_file": "dump_bispec35",
            "snad_file":   "dump_derbispec35",
            "atom_species": atom_species_35,
            "energy": energy_35,
            "forces": forces35,
            "force_units": force_units_35,
        },
        {
            "bispec_file": "dump_bispec36",
            "snad_file":   "dump_derbispec36",
            "atom_species": atom_species_36,
            "energy": energy_36,
            "forces": forces36,
            "force_units": force_units_36,
        },
        {
            "bispec_file": "dump_bispec37",
            "snad_file":   "dump_derbispec37",
            "atom_species": atom_species_37,
            "energy": energy_37,
            "forces": forces37,
            "force_units": force_units_37,
        },
        {
            "bispec_file": "dump_bispec38",
            "snad_file":   "dump_derbispec38",
            "atom_species": atom_species_38,
            "energy": energy_38,
            "forces": forces38,
            "force_units": force_units_38,
        },
        {
            "bispec_file": "dump_bispec39",
            "snad_file":   "dump_derbispec39",
            "atom_species": atom_species_39,
            "energy": energy_39,
            "forces": forces39,
            "force_units": force_units_39,
        },
        {
            "bispec_file": "dump_bispec40",
            "snad_file":   "dump_derbispec40",
            "atom_species": atom_species_40,
            "energy": energy_40,
            "forces": forces40,
            "force_units": force_units_40,
        },
        {
            "bispec_file": "dump_bispec41",
            "snad_file":   "dump_derbispec41",
            "atom_species": atom_species_41,
            "energy": energy_41,
            "forces": forces41,
            "force_units": force_units_41,
        },
        {
            "bispec_file": "dump_bispec42",
            "snad_file":   "dump_derbispec42",
            "atom_species": atom_species_42,
            "energy": energy_42,
            "forces": forces42,
            "force_units": force_units_42,
        },
        {
            "bispec_file": "dump_bispec43",
            "snad_file":   "dump_derbispec43",
            "atom_species": atom_species_43,
            "energy": energy_43,
            "forces": forces43,
            "force_units": force_units_43,
        },
        {
            "bispec_file": "dump_bispec44",
            "snad_file":   "dump_derbispec44",
            "atom_species": atom_species_44,
            "energy": energy_44,
            "forces": forces44,
            "force_units": force_units_44,
        },
        {
            "bispec_file": "dump_bispec45",
            "snad_file":   "dump_derbispec45",
            "atom_species": atom_species_45,
            "energy": energy_45,
            "forces": forces45,
            "force_units": force_units_45,
        },
        {
            "bispec_file": "dump_bispec46",
            "snad_file":   "dump_derbispec46",
            "atom_species": atom_species_46,
            "energy": energy_46,
            "forces": forces46,
            "force_units": force_units_46,
        },
        {
            "bispec_file": "dump_bispec47",
            "snad_file":   "dump_derbispec47",
            "atom_species": atom_species_47,
            "energy": energy_47,
            "forces": forces47,
            "force_units": force_units_47,
        },
        {
            "bispec_file": "dump_bispec48",
            "snad_file":   "dump_derbispec48",
            "atom_species": atom_species_48,
            "energy": energy_48,
            "forces": forces48,
            "force_units": force_units_48,
        },
        {
            "bispec_file": "dump_bispec49",
            "snad_file":   "dump_derbispec49",
            "atom_species": atom_species_49,
            "energy": energy_49,
            "forces": forces49,
            "force_units": force_units_49,
        },
        {
            "bispec_file": "dump_bispec50",
            "snad_file":   "dump_derbispec50",
            "atom_species": atom_species_50,
            "energy": energy_50,
            "forces": forces50,
            "force_units": force_units_50,
        },
        {
            "bispec_file": "dump_bispec51",
            "snad_file":   "dump_derbispec51",
            "atom_species": atom_species_51,
            "energy": energy_51,
            "forces": forces51,
            "force_units": force_units_51,
        },
        {
            "bispec_file": "dump_bispec52",
            "snad_file":   "dump_derbispec52",
            "atom_species": atom_species_52,
            "energy": energy_52,
            "forces": forces52,
            "force_units": force_units_52,
        },
        {
            "bispec_file": "dump_bispec53",
            "snad_file":   "dump_derbispec53",
            "atom_species": atom_species_53,
            "energy": energy_53,
            "forces": forces53,
            "force_units": force_units_53,
        },
        {
            "bispec_file": "dump_bispec54",
            "snad_file":   "dump_derbispec54",
            "atom_species": atom_species_54,
            "energy": energy_54,
            "forces": forces54,
            "force_units": force_units_54,
        },
        {
            "bispec_file": "dump_bispec55",
            "snad_file":   "dump_derbispec55",
            "atom_species": atom_species_55,
            "energy": energy_55,
            "forces": forces55,
            "force_units": force_units_55,
        },
        {
            "bispec_file": "dump_bispec56",
            "snad_file":   "dump_derbispec56",
            "atom_species": atom_species_56,
            "energy": energy_56,
            "forces": forces56,
            "force_units": force_units_56,
        },
        {
            "bispec_file": "dump_bispec57",
            "snad_file":   "dump_derbispec57",
            "atom_species": atom_species_57,
            "energy": energy_57,
            "forces": forces57,
            "force_units": force_units_57,
        }
        ]

# Species block order in SNAD (unique species, NOT per-atom list)
    species_list = ["C", "H"]

    # Regularization and squared weights
    lambda_en = 0.001
    w_energy_sq = math.sqrt(3.0 * 12.0)
    w_forces_sq = 1.0

    # ---- Fit energies + forces ----
    beta, model, normalizer, K = fit_ridge_energy_forces(
        training_set,
        species_list,
        lambda_en=lambda_en,
        w_energy_sq=w_energy_sq,
        w_forces_sq=w_forces_sq,
        lammps_type_order=None,  # or e.g. ["C","H"] if that’s the SNAD block order
    )

    # Pretty print β
    col_names = [f"C{k}" for k in range(K)]
    beta_df = pd.DataFrame(beta, index=species_list, columns=col_names)
    print("\n=== Learned beta (per species × feature) ===")
    pd.set_option("display.precision", 11)
    print(beta_df)

    # Overall metrics (energies + forces)
    metrics = evaluate_training_set(training_set, species_list, beta, normalizer, K)
    print(f"\nRMSE (energy) [kcal/mol]: {metrics['rmse_energy']:.11f}")
    print(f"RMSE (force components) [kcal/mol/Å]: {metrics['rmse_force']:.11f}")

    # Baseline reported for clarity; model has no intercept
    print("\nIntercept (global baseline): 0.00000000000")

    # Per-structure energy predictions (target vs. pred)
    print("\n=== Training set predictions ===")
    y_true, y_pred, n_atoms_list = [], [], []
    for entry in training_set:
        B = read_sna_dump_add_c0(entry["bispec_file"])
        predE = predict_energy(B, entry["atom_species"], species_list, beta, normalizer)
        E = float(entry["energy"])
        print(f"{entry['bispec_file']}: E_target={E:.11f}, E_pred={predE:.11f}, residual={E - predE:.11f}")
        y_true.append(E)
        y_pred.append(predE)
        n_atoms_list.append(len(entry["atom_species"]))

    rmse_total, rmse_per_atom = _compute_rmse_energy(y_true, y_pred, n_atoms_list)
    print(f"\n=== Training set RMSE (total energy): {rmse_total:.11f}")
    print(f"=== Training set RMSE per atom : {rmse_per_atom:.11f}")

    # Incremental RMSEs (with just one structure, it's a short table)
    print("\n=== Incremental RMSEs (ENERGIES + FORCES, fit on first k structures) ===")
    lc_df = incremental_rmse_over_structures_energy_forces(
        training_set,
        species_list,
        lambda_en=lambda_en,
        w_energy_sq=w_energy_sq,
        w_forces_sq=w_forces_sq,
        lammps_type_order=None,
    )
    for row in lc_df.itertuples():
        print(
            f"k={row.k:3d}: "
            f"RMSE_energy={row.rmse_energy:.11f}, "
            f"RMSE_energy_per_atom={row.rmse_energy_per_atom:.11f}, "
            f"RMSE_force={row.rmse_force:.11f}"
        )

    lc_df.to_csv("learning_curve_train_EF.csv", index=False, float_format="%.11f")
    print("\nSaved incremental RMSEs to learning_curve_train_EF.csv")



