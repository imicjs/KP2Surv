"""Clinical loading, expression alignment, WSI sampling, and datasets."""

from __future__ import annotations

from .common import *

class ClinicalDataLoader:
    @staticmethod
    def load(clinical_path: str, sheet_name: str = 'clinical') -> Dict[str, Dict[str, float]]:
        print(f"\n[Data] Loading clinical data: {clinical_path}")
        if not os.path.exists(clinical_path):
            raise FileNotFoundError(f"Clinical file not found: {clinical_path}")

        try:
            if clinical_path.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(clinical_path, sheet_name=sheet_name)
            else:
                df = pd.read_csv(clinical_path, sep='\t' if clinical_path.endswith('.tsv') else ',')
        except Exception:
            df = pd.read_excel(clinical_path) if clinical_path.endswith(('.xls', '.xlsx')) else pd.read_csv(clinical_path)

        df.columns = [str(c).strip() for c in df.columns]
        
        id_col, vital_col, death_col, follow_col = None, None, None, None
        for col in df.columns:
            c = col.lower().replace(' ', '_')
            if c in ['case_id', 'case_submitter_id', 'submitter_id', 'bcr_patient_barcode', 'patient_id', 'patient']:
                id_col = col
            if c in ['vital_status', 'vital', 'os_status', 'event']: vital_col = col
            if c in ['days_to_death', 'death_days', 'os_time']: death_col = col
            if c in ['days_to_last_followup', 'days_to_last_follow_up', 'followup_days', 'follow_up_days']: follow_col = col

        if not vital_col: raise ValueError("Vital status column missing")
        if id_col is None:
            id_col = df.columns[0]

        clinical_dict = {}
        for _, row in df.iterrows():
            raw_id = str(row[id_col]).strip()
            case_id = '-'.join(raw_id.split('-')[:3])
            
            status_str = str(row[vital_col]).strip().lower()
            if status_str in ['dead', 'deceased', '1', 'true'] or 'deceased' in status_str or 'dead' in status_str:
                event = 1
            elif status_str in ['alive', 'living', '0', 'false'] or 'living' in status_str or 'alive' in status_str:
                event = 0
            else: continue

            d_days = float(row[death_col]) if death_col and pd.notna(row[death_col]) else np.nan
            f_days = float(row[follow_col]) if follow_col and pd.notna(row[follow_col]) else np.nan

            if event == 1 and d_days > 0: timev = d_days
            else: timev = max([d for d in [d_days, f_days] if pd.notna(d) and d > 0], default=0)

            if timev > 0:
                if case_id in clinical_dict:
                    if timev > clinical_dict[case_id]['time']:
                        clinical_dict[case_id] = {'time': timev, 'event': int(event)}
                else:
                    clinical_dict[case_id] = {'time': timev, 'event': int(event)}

        print(f"    Valid patients: {len(clinical_dict)}")
        return clinical_dict

@dataclass
class PathwayGeneMask:
    mask: torch.Tensor
    pathway_names: Optional[List[str]] = None
    gene_names: Optional[List[str]] = None
    @property
    def P(self) -> int: return int(self.mask.size(0))
    @property
    def G(self) -> int: return int(self.mask.size(1))


def index_case_files_lexicographically(filenames: List[str]) -> Dict[str, str]:
    """Select the lexicographically first filename for each case."""
    case_to_file: Dict[str, str] = {}
    for filename in sorted(str(name) for name in filenames):
        case_id = extract_case_id(os.path.splitext(filename)[0])
        case_to_file.setdefault(case_id, filename)
    return case_to_file

def load_pathway_gene_mask(mask_path: str, device: torch.device) -> PathwayGeneMask:
    if mask_path.endswith('.npz'):
        data = np.load(mask_path, allow_pickle=True)
        mask = torch.from_numpy(data['mask'].astype(np.float32)).to(device)
        pathway_names = None
        gene_names = None
        if 'pathway_names' in data.files:
            pathway_names = [str(x) for x in data['pathway_names'].tolist()]
        if 'gene_names' in data.files:
            gene_names = [str(x) for x in data['gene_names'].tolist()]
        return PathwayGeneMask(mask=mask, pathway_names=pathway_names, gene_names=gene_names)
    else:
        df = pd.read_csv(mask_path, index_col=0)
        return PathwayGeneMask(
            mask=torch.from_numpy(df.values.astype(np.float32)).to(device),
            pathway_names=df.index.astype(str).tolist(),
            gene_names=df.columns.astype(str).tolist()
        )

def load_expression_table(expr_path: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(expr_path, index_col=0)
    df.index = df.index.map(lambda x: '-'.join(str(x).split('-')[:3]))
    return df, df.columns.astype(str).tolist()

def align_expression_to_mask(expr_df: pd.DataFrame, mask_obj: PathwayGeneMask) -> Tuple[pd.DataFrame, List[str]]:
    if mask_obj.gene_names:
        common = [g for g in mask_obj.gene_names if g in expr_df.columns]
        if len(common) < len(mask_obj.gene_names):
            for g in mask_obj.gene_names:
                if g not in expr_df.columns: expr_df[g] = 0.0
        return expr_df[mask_obj.gene_names], mask_obj.gene_names
    return expr_df, expr_df.columns.tolist()

def compute_train_gene_stats(expr_df: pd.DataFrame, train_ids: List[str], gene_order: List[str]):
    valid_ids = [i for i in train_ids if i in expr_df.index]
    sub = expr_df.loc[valid_ids, gene_order]
    mu = sub.mean(axis=0).values.astype(np.float32)
    sd = sub.std(axis=0, ddof=0).values.astype(np.float32)
    sd[sd < 1e-6] = 1.0
    return mu, sd

# 2. WSI Sampling Helper
# -----------------------------------------------------------------------------
def sample_wsi_nodes_fixed_n(x: torch.Tensor, N: int = 500, K: int = 50, seed: int = 42) -> torch.Tensor:
    n = x.size(0)
    if n <= N: return torch.arange(n, device=x.device)
    
    if not HAVE_SKLEARN or n < K:
        rng = np.random.RandomState(seed)
        return torch.from_numpy(rng.choice(n, size=N, replace=False)).to(x.device).long()
    
    feats = x.detach().cpu().numpy()
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-6)
    km = MiniBatchKMeans(n_clusters=K, batch_size=1024, n_init=3, random_state=seed)
    labels = km.fit_predict(feats)
    
    rng = np.random.RandomState(seed)
    picked = []
    per_cluster = max(1, N // K)
    for c in range(K):
        ids = np.where(labels == c)[0]
        if len(ids) == 0: continue
        take = min(per_cluster, len(ids))
        picked.extend(rng.choice(ids, size=take, replace=False).tolist())
    
    if len(picked) < N:
        remain = list(set(range(n)) - set(picked))
        if remain:
            picked.extend(rng.choice(remain, size=N-len(picked), replace=False).tolist())
            
    return torch.tensor(picked[:N], device=x.device).long()


def _normalize01(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if v.numel() == 0:
        return v
    v_min = v.min()
    v_max = v.max()
    return (v - v_min) / (v_max - v_min + eps)


def segment_normalize01(v: torch.Tensor, batch_index: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize values to [0,1] within each sample segment.
    Prevents cross-sample leakage when batch_size > 1.
    """
    if v.numel() == 0:
        return v
    if batch_index.numel() != v.numel():
        raise RuntimeError(f"segment_normalize01 size mismatch: v={v.numel()} batch={batch_index.numel()}")
    out = torch.zeros_like(v)
    for bid in torch.unique(batch_index, sorted=True):
        m = batch_index == bid
        vals = v[m]
        v_min = vals.min()
        v_max = vals.max()
        out[m] = (vals - v_min) / (v_max - v_min + eps)
    return out


def _compute_wsi_sampling_scores(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (prior_hot_score, hard_replay_score, artifact_score), all in [0,1].
    These are weak continuous heuristics for mixed sampling.
    """
    if x.numel() == 0:
        z = x.new_zeros((0,))
        return z, z, z

    feat_norm = x.norm(p=2, dim=-1)
    centered = x - x.mean(dim=0, keepdim=True)
    hetero = centered.pow(2).mean(dim=-1).sqrt()
    abs_mean = x.abs().mean(dim=-1)
    abs_std = x.std(dim=-1, unbiased=False)

    prior_hot = 0.5 * _normalize01(hetero) + 0.3 * _normalize01(feat_norm) + 0.2 * _normalize01(abs_mean)
    hard_replay = 0.6 * _normalize01(hetero * (feat_norm + 1e-6)) + 0.4 * _normalize01(abs_std)

    # Artifact proxy: low local variation + unstable activation magnitude.
    low_var = 1.0 - _normalize01(abs_std)
    extreme_mag = _normalize01((abs_mean - abs_mean.mean()).abs())
    artifact = torch.clamp(0.6 * low_var + 0.4 * extreme_mag, min=0.0, max=1.0)
    return prior_hot, hard_replay, artifact


def sample_wsi_nodes_mixed(
    x: torch.Tensor,
    N: int = 500,
    K: int = 50,
    seed: int = 42,
    coverage_ratio: float = 0.50,
    prior_hot_ratio: float = 0.25,
    hard_replay_ratio: float = 0.25,
    artifact_cap_ratio: float = 0.15,
) -> torch.Tensor:
    """
    Mixed WSI node-sampling policy:
    50% coverage + 25% prior-hot + 25% hard-replay (by default),
    then cap artifact-dominant nodes.
    """
    n = x.size(0)
    if n <= N:
        return torch.arange(n, device=x.device)

    prior_hot_ratio = max(0.0, float(prior_hot_ratio))
    hard_replay_ratio = max(0.0, float(hard_replay_ratio))
    coverage_ratio = max(0.0, float(coverage_ratio))
    total_ratio = prior_hot_ratio + hard_replay_ratio + coverage_ratio
    if total_ratio <= 0.0:
        coverage_ratio = 1.0
        total_ratio = 1.0

    prior_hot_ratio /= total_ratio
    hard_replay_ratio /= total_ratio
    coverage_ratio /= total_ratio

    n_cov = int(round(N * coverage_ratio))
    n_prior = int(round(N * prior_hot_ratio))
    n_hard = max(0, N - n_cov - n_prior)

    # 1) Coverage nodes
    cov_idx = sample_wsi_nodes_fixed_n(x, N=max(1, n_cov), K=K, seed=seed)
    selected = set(cov_idx.detach().cpu().tolist())

    prior_hot, hard_replay, artifact = _compute_wsi_sampling_scores(x)
    all_idx = torch.arange(n, device=x.device)

    def _take_topk(score: torch.Tensor, k: int, exclude: set) -> List[int]:
        if k <= 0:
            return []
        order = torch.argsort(score, descending=True).detach().cpu().tolist()
        out = []
        for idx in order:
            if idx not in exclude:
                out.append(int(idx))
            if len(out) >= k:
                break
        return out

    # 2) Prior-hot nodes
    prior_idx = _take_topk(prior_hot, n_prior, selected)
    selected.update(prior_idx)

    # 3) Hard-replay nodes
    hard_idx = _take_topk(hard_replay, n_hard, selected)
    selected.update(hard_idx)

    # Fill if not enough
    if len(selected) < N:
        rng = np.random.RandomState(seed)
        remain = [i for i in range(n) if i not in selected]
        if remain:
            take = min(N - len(selected), len(remain))
            selected.update(rng.choice(remain, size=take, replace=False).tolist())

    # Cap artifact-heavy nodes in selected subset
    selected_list = list(selected)[:N]
    if selected_list:
        artifact_cap_ratio = float(np.clip(artifact_cap_ratio, 0.0, 1.0))
        max_art = int(round(N * artifact_cap_ratio))
        if max_art >= 0:
            selected_tensor = torch.tensor(selected_list, device=x.device, dtype=torch.long)
            art_scores_sel = artifact[selected_tensor]
            # artifact positives: top 20% artifact score among selected
            if art_scores_sel.numel() > 5:
                thr = torch.quantile(art_scores_sel, 0.8)
                art_mask = art_scores_sel >= thr
                art_count = int(art_mask.sum().item())
                if art_count > max_art:
                    # remove artifact nodes with low hard score first
                    art_indices = selected_tensor[art_mask]
                    art_hard = hard_replay[art_indices]
                    remove_order = torch.argsort(art_hard, descending=False).detach().cpu().tolist()
                    remove_num = art_count - max_art
                    to_remove = set(int(art_indices[i].item()) for i in remove_order[:remove_num])

                    non_selected = [i for i in range(n) if i not in selected and i not in to_remove]
                    add_candidates = sorted(non_selected, key=lambda j: float(hard_replay[j]), reverse=True)
                    add_take = add_candidates[:remove_num]
                    selected = {i for i in selected if i not in to_remove}
                    selected.update(add_take)
                    selected_list = list(selected)[:N]

    if len(selected_list) < N:
        rng = np.random.RandomState(seed + 7)
        remain = [i for i in range(n) if i not in selected]
        if remain:
            take = min(N - len(selected_list), len(remain))
            selected_list.extend(rng.choice(remain, size=take, replace=False).tolist())

    return torch.tensor(selected_list[:N], device=x.device, dtype=torch.long)


def _compute_hyperedge_sampling_scores(data: GeomData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aggregate existing node-level sampling heuristics to WSI hyperedges.
    Returns (prior_hot, hard_replay, artifact, edge_size) with one value per hyperedge.
    """
    he = data.hyperedge_index
    if he is None or he.numel() == 0:
        z = data.x.new_zeros((0,))
        return z, z, z, z

    x = data.x
    node_ids = he[0].long()
    he_ids = he[1].long()
    num_hyperedges = int(he_ids.max().item()) + 1

    node_prior, node_hard, node_artifact = _compute_wsi_sampling_scores(x)
    edge_size = scatter_add(
        torch.ones_like(he_ids, dtype=x.dtype),
        he_ids,
        dim=0,
        dim_size=num_hyperedges,
    ).clamp_min(1.0)

    prior_hot = scatter_mean(node_prior[node_ids], he_ids, dim=0, dim_size=num_hyperedges)
    hard_replay = scatter_mean(node_hard[node_ids], he_ids, dim=0, dim_size=num_hyperedges)
    artifact = scatter_mean(node_artifact[node_ids], he_ids, dim=0, dim_size=num_hyperedges)

    size_score = _normalize01(edge_size)
    prior_hot = _normalize01(0.85 * prior_hot + 0.15 * size_score)
    hard_replay = _normalize01(0.80 * hard_replay + 0.20 * size_score)
    artifact = artifact.clamp(0.0, 1.0)
    return prior_hot, hard_replay, artifact, edge_size


def sample_wsi_hyperedges_mixed(
    data: GeomData,
    N: int = 500,
    K: int = 50,
    seed: int = 42,
    coverage_ratio: float = 0.50,
    prior_hot_ratio: float = 0.25,
    hard_replay_ratio: float = 0.25,
    artifact_cap_ratio: float = 0.15,
) -> torch.Tensor:
    """
    Hyperedge-centric mixed sampler.
    Selects WSI hyperedges first, then downstream induction keeps their incident nodes
    under the node budget. This preserves pathology structures better than node-first
    sampling when the model learns hyperedge-level importance.
    """
    he = data.hyperedge_index
    if he is None or he.numel() == 0:
        return torch.empty((0,), device=data.x.device, dtype=torch.long)

    device = data.x.device
    prior_hot, hard_replay, artifact, edge_size = _compute_hyperedge_sampling_scores(data)
    num_hyperedges = int(prior_hot.numel())
    if num_hyperedges <= 0:
        return torch.empty((0,), device=device, dtype=torch.long)

    mean_size = float(edge_size.detach().mean().clamp_min(1.0).cpu().item())
    target_edges = int(math.ceil(max(1, int(N)) / max(1.0, mean_size)))
    target_edges = max(1, min(num_hyperedges, target_edges))

    coverage_ratio = max(0.0, float(coverage_ratio))
    prior_hot_ratio = max(0.0, float(prior_hot_ratio))
    hard_replay_ratio = max(0.0, float(hard_replay_ratio))
    total_ratio = coverage_ratio + prior_hot_ratio + hard_replay_ratio
    if total_ratio <= 0.0:
        coverage_ratio, total_ratio = 1.0, 1.0
    coverage_ratio /= total_ratio
    prior_hot_ratio /= total_ratio
    hard_replay_ratio /= total_ratio

    n_cov = int(round(target_edges * coverage_ratio))
    n_prior = int(round(target_edges * prior_hot_ratio))
    n_hard = max(0, target_edges - n_cov - n_prior)

    node_ids = he[0].long()
    he_ids = he[1].long()
    he_feat = scatter_mean(data.x[node_ids].float(), he_ids, dim=0, dim_size=num_hyperedges)

    if num_hyperedges <= target_edges:
        selected = set(range(num_hyperedges))
    else:
        cov_idx = sample_wsi_nodes_fixed_n(
            he_feat,
            N=max(1, n_cov),
            K=max(1, min(K, max(1, n_cov))),
            seed=seed,
        )
        selected = set(cov_idx.detach().cpu().tolist())

        def _take_topk(score: torch.Tensor, k: int, exclude: Set[int]) -> List[int]:
            if k <= 0:
                return []
            order = torch.argsort(score, descending=True).detach().cpu().tolist()
            out: List[int] = []
            for idx in order:
                if int(idx) not in exclude:
                    out.append(int(idx))
                if len(out) >= k:
                    break
            return out

        prior_idx = _take_topk(prior_hot, n_prior, selected)
        selected.update(prior_idx)
        hard_idx = _take_topk(hard_replay, n_hard, selected)
        selected.update(hard_idx)

        if len(selected) < target_edges:
            rng = np.random.RandomState(seed)
            remain = [i for i in range(num_hyperedges) if i not in selected]
            if remain:
                take = min(target_edges - len(selected), len(remain))
                selected.update(rng.choice(remain, size=take, replace=False).tolist())

    selected_list = list(selected)[:target_edges]
    if selected_list:
        artifact_cap_ratio = float(np.clip(artifact_cap_ratio, 0.0, 1.0))
        max_art = int(round(target_edges * artifact_cap_ratio))
        selected_tensor = torch.tensor(selected_list, device=device, dtype=torch.long)
        art_scores_sel = artifact[selected_tensor]
        if art_scores_sel.numel() > 5 and max_art >= 0:
            thr = torch.quantile(art_scores_sel, 0.8)
            art_mask = art_scores_sel >= thr
            art_count = int(art_mask.sum().item())
            if art_count > max_art:
                art_indices = selected_tensor[art_mask]
                art_hard = hard_replay[art_indices]
                remove_order = torch.argsort(art_hard, descending=False).detach().cpu().tolist()
                remove_num = art_count - max_art
                to_remove = set(int(art_indices[i].item()) for i in remove_order[:remove_num])
                add_candidates = [
                    i for i in torch.argsort(hard_replay, descending=True).detach().cpu().tolist()
                    if int(i) not in selected and int(i) not in to_remove
                ]
                selected = {i for i in selected if i not in to_remove}
                selected.update(int(i) for i in add_candidates[:remove_num])
                selected_list = list(selected)[:target_edges]

    return torch.tensor(selected_list[:target_edges], device=device, dtype=torch.long)


def _select_nodes_for_hyperedges(
    data: GeomData,
    selected_hyperedges: torch.Tensor,
    max_nodes: Optional[int],
    seed: int = 42,
) -> torch.Tensor:
    he = data.hyperedge_index
    device = data.x.device
    if he is None or he.numel() == 0 or selected_hyperedges.numel() == 0:
        return sample_wsi_nodes_mixed(data.x, N=max_nodes or data.x.size(0), seed=seed)

    num_hyperedges = infer_num_hyperedges(he)
    edge_mask_lookup = torch.zeros(num_hyperedges, dtype=torch.bool, device=device)
    edge_mask_lookup[selected_hyperedges.long().clamp(min=0, max=max(0, num_hyperedges - 1))] = True
    incidence_mask = edge_mask_lookup[he[1].long()]
    covered_nodes = torch.unique(he[0].long()[incidence_mask])
    if max_nodes is None or int(max_nodes) <= 0 or covered_nodes.numel() <= int(max_nodes):
        return covered_nodes

    max_nodes = int(max_nodes)
    x_cov = data.x[covered_nodes]
    n_cov = max(1, int(round(max_nodes * 0.40)))
    coverage_local = sample_wsi_nodes_fixed_n(
        x_cov,
        N=min(n_cov, covered_nodes.numel()),
        K=max(1, min(50, n_cov)),
        seed=seed,
    )
    selected_local = set(coverage_local.detach().cpu().tolist())

    node_prior, node_hard, node_artifact = _compute_wsi_sampling_scores(data.x)
    selected_inc = scatter_add(
        incidence_mask.to(data.x.dtype),
        he[0].long(),
        dim=0,
        dim_size=int(data.num_nodes) if data.num_nodes is not None else int(data.x.size(0)),
    )
    score = (
        0.35 * node_prior[covered_nodes]
        + 0.35 * node_hard[covered_nodes]
        + 0.20 * _normalize01(selected_inc[covered_nodes])
        - 0.10 * node_artifact[covered_nodes]
    )
    order = torch.argsort(score, descending=True).detach().cpu().tolist()
    for local_idx in order:
        selected_local.add(int(local_idx))
        if len(selected_local) >= max_nodes:
            break

    chosen_local = torch.tensor(list(selected_local)[:max_nodes], device=device, dtype=torch.long)
    return covered_nodes[chosen_local]


def induce_sub_hypergraph(data: GeomData, node_idx: torch.Tensor) -> HypergraphData:
    device = data.x.device
    node_idx = torch.unique(node_idx)
    node_idx_sorted, _ = torch.sort(node_idx)

    def _build_subgraph(hyperedge_index: torch.Tensor, selected_hyperedges: Optional[torch.Tensor] = None) -> HypergraphData:
        out = HypergraphData(
            x=data.x[node_idx_sorted],
            hyperedge_index=hyperedge_index,
            num_hyperedges=infer_num_hyperedges(hyperedge_index),
        )
        if hasattr(data, 'pos') and data.pos is not None:
            out.pos = data.pos[node_idx_sorted]
        if hasattr(data, 'centroid') and data.centroid is not None:
            out.centroid = data.centroid[node_idx_sorted]
        if hasattr(data, 'patch_classify_type') and data.patch_classify_type is not None:
            try:
                out.patch_classify_type = data.patch_classify_type[node_idx_sorted]
            except Exception:
                out.patch_classify_type = data.patch_classify_type
        if selected_hyperedges is not None:
            if hasattr(data, 'hyperedge_type') and data.hyperedge_type is not None:
                try:
                    out.hyperedge_type = data.hyperedge_type[selected_hyperedges]
                except Exception:
                    pass
            if hasattr(data, 'hyperedge_scale') and data.hyperedge_scale is not None:
                try:
                    out.hyperedge_scale = data.hyperedge_scale[selected_hyperedges]
                except Exception:
                    pass
        out.orig_node_idx = node_idx_sorted
        out.orig_hyperedge_idx = (
            selected_hyperedges.to(device=device, dtype=torch.long)
            if selected_hyperedges is not None
            else torch.empty((0,), device=device, dtype=torch.long)
        )
        return out
    
    total_nodes = int(data.num_nodes) if data.num_nodes is not None else int(data.x.size(0))
    mapping = torch.full((total_nodes,), -1, dtype=torch.long, device=device)
    mapping[node_idx_sorted] = torch.arange(len(node_idx_sorted), device=device)
    
    he = data.hyperedge_index
    if he.numel() == 0:
        return _build_subgraph(torch.empty((2,0), device=device, dtype=torch.long), None)
    
    nodes_in_he = he[0]
    mask = mapping[nodes_in_he] >= 0
    new_src = mapping[nodes_in_he[mask]]
    new_dst = he[1][mask]
    
    if new_src.numel() == 0:
        return _build_subgraph(torch.empty((2,0), device=device, dtype=torch.long), None)

    unique_he, inv_he = torch.unique(new_dst, return_inverse=True)
    new_he_indices = torch.stack([new_src, inv_he], dim=0)
    
    return _build_subgraph(new_he_indices, unique_he.long())


def induce_sub_hypergraph_by_hyperedge(
    data: GeomData,
    hyperedge_idx: torch.Tensor,
    max_nodes: Optional[int] = None,
    seed: int = 42,
) -> HypergraphData:
    """
    Build a sub-hypergraph from selected original hyperedges.
    Hyperedges are selected first; incident nodes are then retained up to max_nodes.
    """
    device = data.x.device
    he = data.hyperedge_index
    if he is None or he.numel() == 0:
        node_idx = sample_wsi_nodes_mixed(data.x, N=max_nodes or data.x.size(0), seed=seed)
        out = induce_sub_hypergraph(data, node_idx)
        out.orig_hyperedge_idx = torch.empty((0,), device=device, dtype=torch.long)
        return out

    num_hyperedges = infer_num_hyperedges(he)
    hyperedge_idx = torch.unique(hyperedge_idx.to(device=device, dtype=torch.long))
    hyperedge_idx = hyperedge_idx[(hyperedge_idx >= 0) & (hyperedge_idx < num_hyperedges)]
    if hyperedge_idx.numel() == 0:
        hyperedge_idx = sample_wsi_hyperedges_mixed(data, N=max_nodes or data.x.size(0), seed=seed)

    node_idx = _select_nodes_for_hyperedges(data, hyperedge_idx, max_nodes=max_nodes, seed=seed)
    node_idx = torch.unique(node_idx.to(device=device, dtype=torch.long))
    node_idx_sorted, _ = torch.sort(node_idx)

    total_nodes = int(data.num_nodes) if data.num_nodes is not None else int(data.x.size(0))
    mapping = torch.full((total_nodes,), -1, dtype=torch.long, device=device)
    mapping[node_idx_sorted] = torch.arange(node_idx_sorted.numel(), device=device)

    edge_lookup = torch.zeros(num_hyperedges, dtype=torch.bool, device=device)
    edge_lookup[hyperedge_idx] = True
    incidence_mask = edge_lookup[he[1].long()] & (mapping[he[0].long()] >= 0)
    if incidence_mask.sum().item() == 0:
        return induce_sub_hypergraph(data, node_idx_sorted)

    old_src = he[0].long()[incidence_mask]
    old_dst = he[1].long()[incidence_mask]
    new_src = mapping[old_src]
    unique_he, inv_he = torch.unique(old_dst, return_inverse=True)
    new_he_indices = torch.stack([new_src, inv_he], dim=0)

    out = HypergraphData(
        x=data.x[node_idx_sorted],
        hyperedge_index=new_he_indices,
        num_hyperedges=infer_num_hyperedges(new_he_indices),
    )
    if hasattr(data, 'pos') and data.pos is not None:
        out.pos = data.pos[node_idx_sorted]
    if hasattr(data, 'centroid') and data.centroid is not None:
        out.centroid = data.centroid[node_idx_sorted]
    if hasattr(data, 'patch_classify_type') and data.patch_classify_type is not None:
        try:
            out.patch_classify_type = data.patch_classify_type[node_idx_sorted]
        except Exception:
            out.patch_classify_type = data.patch_classify_type
    if hasattr(data, 'tumor_prob') and data.tumor_prob is not None:
        try:
            out.tumor_prob = data.tumor_prob[node_idx_sorted]
        except Exception:
            pass
    if hasattr(data, 'hyperedge_type') and data.hyperedge_type is not None:
        try:
            out.hyperedge_type = data.hyperedge_type[unique_he]
        except Exception:
            pass
    if hasattr(data, 'hyperedge_scale') and data.hyperedge_scale is not None:
        try:
            out.hyperedge_scale = data.hyperedge_scale[unique_he]
        except Exception:
            pass

    out.orig_node_idx = node_idx_sorted
    out.orig_hyperedge_idx = unique_he.to(device=device, dtype=torch.long)
    return out

# -----------------------------------------------------------------------------
# 3. Dataset
# -----------------------------------------------------------------------------
class MultiModalSurvivalDataset(Dataset):
    def __init__(self, wsi_dir, gene_dir, clinical, expr_df, gene_order, 
                 ids=None, wsi_N=500, wsi_K=50, seed=42,
                 use_mixed_sampling: bool = True,
                 use_hyperedge_sampling: bool = True,
                 wsi_cache_dir: str = "",
                 stable_wsi_cache_seed: bool = True,
                 rebuild_wsi_cache: bool = False,
                 wsi_cache_variants: int = 1,
                 wsi_cache_variant_mode: str = "fixed",
                 wsi_cache_variant_index: int = 0,
                 coverage_ratio: float = 0.50,
                 prior_hot_ratio: float = 0.25,
                 hard_replay_ratio: float = 0.25,
                 artifact_cap_ratio: float = 0.15):
        self.wsi_dir = wsi_dir
        self.gene_dir = gene_dir
        self.clinical = clinical
        self.expr_df = expr_df
        self.gene_order = gene_order
        self._expr_df_fast = expr_df.loc[:, gene_order].astype(np.float32, copy=False)
        self._expr_np = self._expr_df_fast.to_numpy(dtype=np.float32, copy=False)
        self._expr_row = {str(cid): int(i) for i, cid in enumerate(self._expr_df_fast.index.tolist())}
        self.wsi_N = wsi_N
        self.wsi_K = wsi_K
        self.seed = seed
        self.use_mixed_sampling = bool(use_mixed_sampling)
        self.use_hyperedge_sampling = bool(use_hyperedge_sampling)
        self.wsi_cache_dir = str(wsi_cache_dir or "")
        self.stable_wsi_cache_seed = bool(stable_wsi_cache_seed)
        self.rebuild_wsi_cache = bool(rebuild_wsi_cache)
        self.wsi_cache_variants = max(1, int(wsi_cache_variants))
        self.wsi_cache_variant_mode = str(wsi_cache_variant_mode or "fixed").lower()
        if self.wsi_cache_variant_mode not in {"fixed", "random"}:
            raise ValueError(f"Unsupported wsi_cache_variant_mode={wsi_cache_variant_mode!r}; use 'fixed' or 'random'.")
        self.wsi_cache_variant_index = int(wsi_cache_variant_index)
        if self.wsi_cache_dir:
            ensure_dir(self.wsi_cache_dir)
        self.coverage_ratio = float(coverage_ratio)
        self.prior_hot_ratio = float(prior_hot_ratio)
        self.hard_replay_ratio = float(hard_replay_ratio)
        self.artifact_cap_ratio = float(artifact_cap_ratio)
        
        common_ids = sorted(set(clinical.keys()) & set(expr_df.index))
        wsi_files = [f for f in sorted(os.listdir(wsi_dir)) if f.endswith('.pt')]
        gene_files = [f for f in sorted(os.listdir(gene_dir)) if f.endswith('.pt')]

        wsi_case_to_file = index_case_files_lexicographically(wsi_files)
        gene_case_to_file = index_case_files_lexicographically(gene_files)
        
        valid_ids = []
        for cid in common_ids:
            w_match = wsi_case_to_file.get(cid)
            g_match = gene_case_to_file.get(cid)
            if w_match and g_match:
                if ids is None or cid in ids:
                    valid_ids.append({
                        'case_id': cid, 
                        'wsi_file': w_match, 
                        'gene_file': g_match,
                        'time': clinical[cid]['time'],
                        'event': clinical[cid]['event'],
                    })
        self.samples = valid_ids
        self._wsi_cache = {}
        self._gene_he_cache = {}

    def __len__(self): return len(self.samples)

    def _choose_cache_variant(self) -> int:
        if self.wsi_cache_variants <= 1:
            return 0
        if self.wsi_cache_variant_mode == "random":
            return random.randint(0, self.wsi_cache_variants - 1)
        return max(0, min(int(self.wsi_cache_variant_index), self.wsi_cache_variants - 1))

    def _sample_seed(self, wsi_file: str, idx: int, cache_variant: int = 0) -> int:
        if self.stable_wsi_cache_seed:
            base = int(self.seed) + stable_hash_int(wsi_file)
        else:
            base = int(self.seed) + int(idx)
        return int(base) + int(cache_variant) * 1000003

    def _memory_cache_key(self, wsi_file: str, sample_seed: int, cache_variant: int) -> Tuple:
        return (
            wsi_file,
            int(sample_seed),
            int(cache_variant),
            int(self.wsi_cache_variants),
            int(self.wsi_N),
            int(self.wsi_K),
            bool(self.use_mixed_sampling),
            bool(self.use_hyperedge_sampling),
            round(float(self.coverage_ratio), 6),
            round(float(self.prior_hot_ratio), 6),
            round(float(self.hard_replay_ratio), 6),
            round(float(self.artifact_cap_ratio), 6),
        )

    def _disk_cache_path(self, wsi_file: str, wsi_path: str, sample_seed: int, cache_variant: int) -> Optional[str]:
        if not self.wsi_cache_dir:
            return None
        try:
            st = os.stat(wsi_path)
            source_sig = {"size": int(st.st_size), "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))}
        except OSError:
            source_sig = {"size": -1, "mtime_ns": -1}
        payload = {
            "version": "kp2surv_hyperedge_subgraph_cache_v1",
            "wsi_file": str(wsi_file),
            "sample_seed": int(sample_seed),
            "cache_variant": int(cache_variant),
            "cache_variants": int(self.wsi_cache_variants),
            "wsi_N": int(self.wsi_N),
            "wsi_K": int(self.wsi_K),
            "use_mixed_sampling": bool(self.use_mixed_sampling),
            "use_hyperedge_sampling": bool(self.use_hyperedge_sampling),
            "coverage_ratio": round(float(self.coverage_ratio), 6),
            "prior_hot_ratio": round(float(self.prior_hot_ratio), 6),
            "hard_replay_ratio": round(float(self.hard_replay_ratio), 6),
            "artifact_cap_ratio": round(float(self.artifact_cap_ratio), 6),
            "source": source_sig,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.blake2b(raw.encode("utf-8"), digest_size=10).hexdigest()
        stem = os.path.splitext(os.path.basename(wsi_file))[0]
        safe_stem = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in stem)
        return os.path.join(self.wsi_cache_dir, f"{safe_stem}_{digest}.pt")

    @staticmethod
    def _save_disk_cache(cache_path: str, data_obj: HypergraphData) -> None:
        parent = os.path.dirname(cache_path)
        if parent:
            ensure_dir(parent)
        tmp = f"{cache_path}.tmp.{os.getpid()}.{random.randint(0, 10**9)}"
        try:
            torch.save(data_obj, tmp)
            os.replace(tmp, cache_path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        cache_variant = self._choose_cache_variant()
        sample_seed = self._sample_seed(item['wsi_file'], idx, cache_variant)
        mem_key = self._memory_cache_key(item['wsi_file'], sample_seed, cache_variant)
        wsi_path = os.path.join(self.wsi_dir, item['wsi_file'])
        disk_cache_path = self._disk_cache_path(item['wsi_file'], wsi_path, sample_seed, cache_variant)

        if FIXED_WSI_SUBGRAPH and mem_key in self._wsi_cache:
            wsi_data = self._wsi_cache[mem_key]
        elif disk_cache_path and (not self.rebuild_wsi_cache) and os.path.isfile(disk_cache_path):
            wsi_data = torch.load(disk_cache_path, map_location='cpu', weights_only=False)
            if FIXED_WSI_SUBGRAPH:
                self._wsi_cache[mem_key] = wsi_data
        else:
            wsi = torch.load(wsi_path, map_location='cpu', weights_only=False)
            if self.use_mixed_sampling:
                if self.use_hyperedge_sampling:
                    hyperedge_idx = sample_wsi_hyperedges_mixed(
                        wsi,
                        N=self.wsi_N,
                        K=self.wsi_K,
                        seed=sample_seed,
                        coverage_ratio=self.coverage_ratio,
                        prior_hot_ratio=self.prior_hot_ratio,
                        hard_replay_ratio=self.hard_replay_ratio,
                        artifact_cap_ratio=self.artifact_cap_ratio,
                    )
                    wsi_data = induce_sub_hypergraph_by_hyperedge(
                        wsi,
                        hyperedge_idx,
                        max_nodes=self.wsi_N,
                        seed=sample_seed,
                    )
                else:
                    node_idx = sample_wsi_nodes_mixed(
                        wsi.x,
                        N=self.wsi_N,
                        K=self.wsi_K,
                        seed=sample_seed,
                        coverage_ratio=self.coverage_ratio,
                        prior_hot_ratio=self.prior_hot_ratio,
                        hard_replay_ratio=self.hard_replay_ratio,
                        artifact_cap_ratio=self.artifact_cap_ratio,
                    )
                    wsi_data = induce_sub_hypergraph(wsi, node_idx)
            else:
                node_idx = sample_wsi_nodes_fixed_n(wsi.x, self.wsi_N, self.wsi_K, sample_seed)
                wsi_data = induce_sub_hypergraph(wsi, node_idx)
            if disk_cache_path:
                try:
                    self._save_disk_cache(disk_cache_path, wsi_data)
                except Exception as e:
                    warnings.warn(f"Failed to write WSI subgraph cache {disk_cache_path}: {e}")
            if FIXED_WSI_SUBGRAPH:
                self._wsi_cache[mem_key] = wsi_data
            
        if item['gene_file'] in self._gene_he_cache:
            gene_he, gene_num_nodes, gene_num_hyperedges = self._gene_he_cache[item['gene_file']]
        else:
            gene = torch.load(os.path.join(self.gene_dir, item['gene_file']), map_location='cpu', weights_only=False)
            gene_he = gene.hyperedge_index
            gene_num_nodes = int(gene.num_nodes) if hasattr(gene, "num_nodes") and gene.num_nodes is not None else (
                int(gene_he[0].max().item()) + 1 if gene_he.numel() > 0 else 0
            )
            gene_num_hyperedges = infer_num_hyperedges(gene_he)
            self._gene_he_cache[item['gene_file']] = (gene_he, gene_num_nodes, gene_num_hyperedges)
             
        expr_row = self._expr_row[item['case_id']]
        expr = torch.from_numpy(self._expr_np[expr_row])
         
        return {
            'wsi': wsi_data,
            'gene_tpl': HypergraphData(
                hyperedge_index=gene_he,
                num_nodes=gene_num_nodes,
                num_hyperedges=gene_num_hyperedges,
            ),
            'expr': expr,
            'time': torch.tensor(item['time'], dtype=torch.float32),
            'event': torch.tensor(item['event'], dtype=torch.float32),
            'case_id': item['case_id'],
            'wsi_file': item['wsi_file'],
            'gene_file': item['gene_file']
        }

def survival_collate(batch):
    wsi_batch = GeomBatch.from_data_list([b['wsi'] for b in batch])
    gene_batch = GeomBatch.from_data_list([b['gene_tpl'] for b in batch])
    return {
        'wsi': wsi_batch,
        'gene_tpl': gene_batch,
        'expr': torch.stack([b['expr'] for b in batch]),
        'time': torch.stack([b['time'] for b in batch]),
        'event': torch.stack([b['event'] for b in batch]),
        'case_id': [b['case_id'] for b in batch],
        'wsi_file': [b['wsi_file'] for b in batch],
        'gene_file': [b['gene_file'] for b in batch]
    }

# -----------------------------------------------------------------------------
