"""Build spatial and feature hypergraphs directly from H5 patch data."""

import os
import argparse
import h5py
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Data as GeomData
from torch_geometric.utils import to_undirected
from collections import deque
import random
import matplotlib.pyplot as plt

try:
    from sklearn.cluster import MiniBatchKMeans, KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn is not installed. KMeans-based feature hyperedges cannot be created.")
    print("Please run: pip install scikit-learn")


def _maybe_int_coords(coords: np.ndarray) -> np.ndarray:
    """Quantize coordinates to integers for stable dictionary lookup."""
    if np.allclose(coords, np.round(coords)):
        return np.round(coords).astype(np.int64)
    return np.round(coords).astype(np.int64)

def estimate_grid_step(coords_int: np.ndarray) -> int:
    """Estimate a positive grid step, falling back to one if necessary."""
    xs = np.unique(coords_int[:, 0])
    ys = np.unique(coords_int[:, 1])
    dx = np.diff(np.sort(xs))
    dy = np.diff(np.sort(ys))
    dx = dx[dx > 0]
    dy = dy[dy > 0]
    candidates = np.concatenate([dx, dy]) if (dx.size + dy.size) > 0 else np.array([], dtype=np.int64)
    if candidates.size == 0:
        return 1
    step = int(np.median(candidates))
    return max(step, 1)


def h5_to_grid_graph(wsi_h5, cls_pth=None):
    """
    Create a four-neighbor grid graph from H5 coordinates and features.

    Returns:
        GeomData: A PyTorch Geometric graph.
    """
    coords = np.array(wsi_h5['coords'])
    features = np.array(wsi_h5['features'])
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected coordinates with shape [N, 2], got {coords.shape}")
    if features.ndim != 2:
        raise ValueError(f"Expected features with shape [N, D], got {features.shape}")
    if coords.shape[0] == 0:
        raise ValueError("The H5 file contains no patch coordinates")
    if coords.shape[0] != features.shape[0]:
        raise ValueError(
            f"Coordinate/feature row mismatch: {coords.shape[0]} != {features.shape[0]}"
        )

    cls = None
    if cls_pth is not None and os.path.exists(cls_pth):
        clss = torch.load(cls_pth)
        if np.array_equal(coords, np.array(clss.centroid)):
            cls = torch.Tensor(clss.patch_classify_type)
        else:
            print(f"  - Warning: coordinates mismatch in {os.path.basename(cls_pth)}")

    coords_int = _maybe_int_coords(coords)

    step = estimate_grid_step(coords_int)

    coord_to_idx = {tuple(coord): i for i, coord in enumerate(coords_int)}

    edge_list = []
    for i, (x, y) in enumerate(coords_int):
        neighbors_coords = [
            (x, y + step),
            (x, y - step),
            (x - step, y),
            (x + step, y),
        ]
        for neighbor_coord in neighbors_coords:
            j = coord_to_idx.get(neighbor_coord, None)
            if j is not None:
                edge_list.append([i, j])

    if not edge_list:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        directed_edges = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(directed_edges)

    G = GeomData(
        x=torch.from_numpy(features).float(),
        edge_index=edge_index,
        pos=torch.from_numpy(coords_int).float(),
        y=cls,
    )
    G.centroid = torch.from_numpy(coords_int).float()
    G.patch_classify_type = cls
    G.grid_step = int(step)

    return G


def find_connected_components(graph_data):
    """Return all connected components in a PyG graph."""
    num_nodes = graph_data.num_nodes
    adj = {i: [] for i in range(num_nodes)}
    edge_index = graph_data.edge_index

    for i in range(edge_index.size(1)):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        adj[u].append(v)
        adj[v].append(u)

    visited = set()
    components = []

    for i in range(num_nodes):
        if i not in visited:
            component = []
            q = deque([i])
            visited.add(i)
            while q:
                node = q.popleft()
                component.append(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)
            components.append(component)

    return components


def _build_local_feature_hyperedges_kmeans(
    graph_data,
    target_cluster_size=20,
    k_min=4,
    k_max=32,
    min_he_size=8,
    max_he_size=4096,
    seed=42,
):
    """Build reproducible feature hyperedges with adaptive cluster counts."""
    if not SKLEARN_AVAILABLE:
        return [], [], []

    all_features = graph_data.x
    components = find_connected_components(graph_data)

    hyperedges, he_types, he_scales = [], [], []
    global_cluster_id_offset = 0
    rng = np.random.RandomState(seed)

    for component_nodes in components:
        n = len(component_nodes)
        if n < min_he_size:
            continue

        k = max(k_min, min(k_max, n // max(target_cluster_size, 1)))
        if k <= 1:
            continue

        feats = all_features[component_nodes].detach().cpu().numpy()
        feats_norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-6)

        if n < 2048:
            km = KMeans(n_clusters=k, n_init='auto', random_state=seed)
        else:
            km = MiniBatchKMeans(n_clusters=k, batch_size=1024, n_init=5, random_state=seed)

        try:
            labels = km.fit_predict(feats_norm)
        except ValueError:
            continue

        for c in range(k):
            local_idx = np.where(labels == c)[0]
            if local_idx.size < min_he_size:
                continue

            nodes = [component_nodes[i] for i in local_idx.tolist()]
            if len(nodes) > max_he_size:
                nodes = rng.choice(nodes, size=max_he_size, replace=False).tolist()

            hyperedges.append(nodes)
            he_types.append(1)
            he_scales.append(global_cluster_id_offset + c)

        global_cluster_id_offset += k

    return hyperedges, he_types, he_scales


def _select_centers_binning(coords: np.ndarray, bin_size: int, max_centers: int, seed: int):
    """Select one center per spatial bin, subject to an optional limit."""
    rng = random.Random(seed)
    min_xy = coords.min(axis=0)

    bins = {}
    for i, (x, y) in enumerate(coords):
        bx = int((x - min_xy[0]) // bin_size)
        by = int((y - min_xy[1]) // bin_size)
        bins.setdefault((bx, by), []).append(i)

    centers = [rng.choice(v) for v in bins.values() if len(v) > 0]
    if max_centers is not None and len(centers) > max_centers:
        centers = rng.sample(centers, max_centers)
    return centers


def _build_spatial_hyperedges_khop_from_graph(
    graph_data,
    khops=(1, 2),
    stride=8,
    min_he_size=5,
    max_centers=300,
    seed=42,
):
    """Build k-hop spatial hyperedges around spatially binned centers."""
    num_nodes = graph_data.num_nodes
    coords = graph_data.pos.detach().cpu().numpy()

    adj = {i: [] for i in range(num_nodes)}
    for i in range(graph_data.edge_index.size(1)):
        u, v = graph_data.edge_index[:, i].tolist()
        adj[u].append(v)
        adj[v].append(u)

    grid_step = int(getattr(graph_data, "grid_step", 1))
    bin_size = int(max(1, stride * grid_step))

    centers = _select_centers_binning(coords, bin_size=bin_size, max_centers=max_centers, seed=seed)

    hyperedges, he_types, he_scales = [], [], []
    max_k = max(khops)

    for center_idx in centers:
        q = deque([(center_idx, 0)])
        visited = {center_idx}
        levels = {0: [center_idx]}

        while q:
            curr_node, dist = q.popleft()
            if dist >= max_k:
                continue
            for neighbor in adj[curr_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append((neighbor, dist + 1))
                    levels.setdefault(dist + 1, []).append(neighbor)

        for k in khops:
            ball_nodes = []
            for d in range(k + 1):
                ball_nodes.extend(levels.get(d, []))
            if len(ball_nodes) >= min_he_size:
                hyperedges.append(ball_nodes)
                he_types.append(0)
                he_scales.append(k)

    return hyperedges, he_types, he_scales


def _dedup_and_filter(hyperedges, he_types, he_scales, min_size=3):
    """Deduplicate hyperedges and remove those below the size threshold."""
    unique_hes = {}
    for he, type_val, scale in zip(hyperedges, he_types, he_scales):
        if len(he) < min_size:
            continue
        key = tuple(sorted(he))
        if key not in unique_hes:
            unique_hes[key] = (he, type_val, scale)

    if not unique_hes:
        return [], [], []

    final_hes, final_types, final_scales = zip(*unique_hes.values())
    return list(final_hes), list(final_types), list(final_scales)


def _to_hyperedge_index(hyperedges):
    """Convert a hyperedge list to PyTorch Geometric incidence format."""
    if not hyperedges:
        return torch.empty((2, 0), dtype=torch.long)

    node_indices = [node for he in hyperedges for node in he]
    he_indices = [i for i, he in enumerate(hyperedges) for _ in he]
    return torch.tensor([node_indices, he_indices], dtype=torch.long)


def _subsample_hyperedges(final_hes, final_types, final_scales, max_total, seed=42):
    if len(final_hes) <= max_total:
        return final_hes, final_types, final_scales
    rng = random.Random(seed)
    keep = rng.sample(range(len(final_hes)), max_total)
    keep_set = set(keep)
    hes2 = [h for i, h in enumerate(final_hes) if i in keep_set]
    t2 = [t for i, t in enumerate(final_types) if i in keep_set]
    s2 = [s for i, s in enumerate(final_scales) if i in keep_set]
    return hes2, t2, s2


def graph_to_hypergraph(graph_data, **kwargs):
    """Convert a graph to a size-constrained spatial-feature hypergraph."""
    seed = int(kwargs.get("seed", 42))

    calib_min_total = int(kwargs.get("calib_min_total_hyperedges", 120))
    calib_max_total = int(kwargs.get("calib_max_total_hyperedges", 1500))
    repair_max_attempts = int(kwargs.get("repair_max_attempts", 2))

    khops = kwargs.get('khops', (1, 2))
    spatial_stride = int(kwargs.get('spatial_stride', 8))
    spatial_min_he_size = int(kwargs.get("spatial_min_he_size", 5))
    spatial_max_centers = int(kwargs.get("spatial_max_centers", 300))

    feature_method = kwargs.get('feature_method')
    feature_target_cluster_size = int(kwargs.get("feature_target_cluster_size", 20))
    feature_k_min = int(kwargs.get("feature_k_min", 4))
    feature_k_max = int(kwargs.get("feature_k_max", 32))
    min_feature_he_size = int(kwargs.get('min_feature_he_size', 8))
    max_he_size = int(kwargs.get('max_he_size', 4096))

    def _build_once(ss: int, f_min: int, f_tgt: int, f_kmax: int):
        spatial_hes, spatial_types, spatial_scales = _build_spatial_hyperedges_khop_from_graph(
            graph_data,
            khops=khops,
            stride=ss,
            min_he_size=spatial_min_he_size,
            max_centers=spatial_max_centers,
            seed=seed,
        )

        if feature_method == 'local_kmeans':
            feat_hes, feat_types, feat_scales = _build_local_feature_hyperedges_kmeans(
                graph_data,
                target_cluster_size=f_tgt,
                k_min=feature_k_min,
                k_max=f_kmax,
                min_he_size=f_min,
                max_he_size=max_he_size,
                seed=seed,
            )
        else:
            feat_hes, feat_types, feat_scales = [], [], []

        all_hes = spatial_hes + feat_hes
        all_types = spatial_types + feat_types
        all_scales = spatial_scales + feat_scales
        final_hes, final_types, final_scales = _dedup_and_filter(all_hes, all_types, all_scales, min_size=3)
        return spatial_hes, feat_hes, final_hes, final_types, final_scales

    attempt = 0
    ss = spatial_stride
    f_min = min_feature_he_size
    f_tgt = feature_target_cluster_size
    f_kmax = feature_k_max

    while attempt <= repair_max_attempts:
        spatial_hes, feat_hes, final_hes, final_types, final_scales = _build_once(ss, f_min, f_tgt, f_kmax)
        total = len(final_hes)

        if total == 0 and attempt < repair_max_attempts:
            attempt += 1
            ss = max(1, ss // 2)
            f_min = max(3, f_min // 2)
            f_tgt = max(10, f_tgt - 5)
            f_kmax = min(64, f_kmax + 8)
            continue

        if total < calib_min_total and attempt < repair_max_attempts:
            attempt += 1
            ss = max(1, ss // 2)
            f_min = max(3, f_min // 2)
            f_tgt = max(10, f_tgt - 5)
            f_kmax = min(64, f_kmax + 8)
            continue

        break

    final_hes, final_types, final_scales = _subsample_hyperedges(
        final_hes, final_types, final_scales, max_total=calib_max_total, seed=seed
    )

    hyperedge_index = _to_hyperedge_index(final_hes)
    num_hyperedges = len(final_hes)

    print(
        f"  - Building hyperedges. Done. "
        f"(Spatial: {len(spatial_hes)}, Feature: {len(feat_hes)}) -> Final: {num_hyperedges} "
        f"[attempt={attempt}, ss={ss}, f_min={f_min}, f_tgt={f_tgt}, f_kmax={f_kmax}]",
        flush=True
    )

    hypergraph_G = GeomData(
        x=graph_data.x,
        pos=graph_data.pos,
        y=graph_data.y if hasattr(graph_data, 'y') else None,
        hyperedge_index=hyperedge_index,
        num_hyperedges=num_hyperedges,
        hyperedge_type=torch.tensor(final_types, dtype=torch.long) if num_hyperedges > 0 else torch.empty((0,), dtype=torch.long),
        hyperedge_scale=torch.tensor(final_scales, dtype=torch.long) if num_hyperedges > 0 else torch.empty((0,), dtype=torch.long)
    )

    if hasattr(graph_data, 'centroid'):
        hypergraph_G.centroid = graph_data.centroid
    if hasattr(graph_data, 'patch_classify_type'):
        hypergraph_G.patch_classify_type = graph_data.patch_classify_type

    hypergraph_G.hg_stats = {
        "num_nodes": int(graph_data.num_nodes),
        "num_edges": int(graph_data.edge_index.size(1)),
        "grid_step": int(getattr(graph_data, "grid_step", 1)),
        "spatial_hes": int(len(spatial_hes)),
        "feature_hes": int(len(feat_hes)),
        "final_hes": int(num_hyperedges),
        "attempt": int(attempt),
        "used_spatial_stride": int(ss),
        "used_feature_min_he_size": int(f_min),
        "used_feature_target_cluster_size": int(f_tgt),
        "used_feature_kmax": int(f_kmax),
        "seed": int(seed),
    }

    return hypergraph_G


def h5_to_hypergraph(wsi_h5, cls_pth=None, **kwargs):
    graph_data = h5_to_grid_graph(wsi_h5, cls_pth)
    hypergraph_data = graph_to_hypergraph(graph_data, **kwargs)
    return hypergraph_data


def convert_h5_directory(h5_path, save_path, node_type_path=None, **kwargs):
    if not os.path.isdir(h5_path):
        print(f"Error: Input directory '{h5_path}' does not exist.")
        return

    os.makedirs(save_path, exist_ok=True)
    all_files = [f for f in os.listdir(h5_path) if f.endswith('.h5')]

    pbar = tqdm(all_files, desc="H5 to hypergraph")

    for h5_fname in pbar:
        h5_full_path = os.path.join(h5_path, h5_fname)
        pbar.set_description(f'{h5_fname[:15]} - Converting')

        try:
            with h5py.File(h5_full_path, "r") as wsi_h5:
                if 'coords' not in wsi_h5 or 'features' not in wsi_h5:
                    pbar.set_description(f'{h5_fname[:15]} - Missing Data')
                    print(f"\nFile {h5_fname} missing 'coords' or 'features', skipping.")
                    continue

                h5_fname_base = h5_fname.replace('.h5', '')
                cls_pth = f"{node_type_path}/{h5_fname_base}.pt" if node_type_path else None

                hypergraph_data = h5_to_hypergraph(wsi_h5, cls_pth, **kwargs)

                save_fname = h5_fname.replace('.h5', '.pt')
                torch.save(hypergraph_data, os.path.join(save_path, save_fname))

        except OSError as e:
            pbar.set_description(f'{h5_fname[:15]} - Broken H5')
            print(f"\nCannot read {h5_fname} (possibly corrupted): {e}")
        except Exception as e:
            pbar.set_description(f'{h5_fname[:15]} - Error')
            print(f"\nError processing {h5_fname}: {e}")
            import traceback
            traceback.print_exc()


def visualize_hyperedges(pt_file_path, save_dir, num_to_show=5):
    data = torch.load(pt_file_path)
    feature_he_indices = (data.hyperedge_type == 1).nonzero(as_tuple=True)[0]

    if len(feature_he_indices) == 0:
        print(f"  - No feature hyperedges found in {os.path.basename(pt_file_path)}, skipping.")
        return

    num_to_show = min(num_to_show, len(feature_he_indices))
    selected_he_indices = random.sample(feature_he_indices.tolist(), num_to_show)

    plt.figure(figsize=(10, 10))
    coords = data.pos.numpy()

    plt.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=5, label='Background Nodes')

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'lime']

    for i, he_idx in enumerate(selected_he_indices):
        mask = (data.hyperedge_index[1] == he_idx)
        nodes_in_he = data.hyperedge_index[0][mask]
        he_coords = coords[nodes_in_he]
        color = colors[i % len(colors)]
        plt.scatter(he_coords[:, 0], he_coords[:, 1], c=color, s=5,
                    label=f'Feature HE {i+1} (ID: {he_idx})')

    plt.title(f'Feature Hyperedge Visualization\n{os.path.basename(pt_file_path)}')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    save_name = os.path.basename(pt_file_path).replace('.pt', '_feature_hyperedges.png')
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    print("="*80)
    print(" Direct H5 to Hypergraph Conversion Pipeline")
    print("="*80)

    ap = argparse.ArgumentParser(description="Build manuscript-aligned WSI hypergraphs from H5 patch features.")
    ap.add_argument("--h5_dir", required=True, help="Directory containing input .h5 patch-feature files.")
    ap.add_argument("--out_dir", required=True, help="Directory for output .pt hypergraphs.")
    ap.add_argument("--node_type_dir", default="", help="Optional directory containing patch-type metadata.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--khops", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--spatial_stride", type=int, default=8)
    ap.add_argument("--spatial_min_he_size", type=int, default=5)
    ap.add_argument("--spatial_max_centers", type=int, default=300)
    ap.add_argument("--feature_target_cluster_size", type=int, default=20)
    ap.add_argument("--feature_k_min", type=int, default=4)
    ap.add_argument("--feature_k_max", type=int, default=32)
    ap.add_argument("--min_feature_he_size", type=int, default=8)
    ap.add_argument("--max_he_size", type=int, default=4096)
    ap.add_argument("--calib_min_total_hyperedges", type=int, default=120)
    ap.add_argument("--calib_max_total_hyperedges", type=int, default=1500)
    ap.add_argument("--repair_max_attempts", type=int, default=2)
    ap.add_argument("--visualize_count", type=int, default=0, help="Number of output slides to visualize; 0 disables visualization.")
    args = ap.parse_args()

    h5_path = args.h5_dir
    final_save_path = args.out_dir
    node_type_path = args.node_type_dir or None

    hypergraph_params = {
        "seed": args.seed,

        "khops": tuple(args.khops),
        "spatial_stride": args.spatial_stride,
        "spatial_min_he_size": args.spatial_min_he_size,
        "spatial_max_centers": args.spatial_max_centers,

        "feature_method": "local_kmeans",
        "feature_target_cluster_size": args.feature_target_cluster_size,
        "feature_k_min": args.feature_k_min,
        "feature_k_max": args.feature_k_max,
        "min_feature_he_size": args.min_feature_he_size,
        "max_he_size": args.max_he_size,

        "calib_min_total_hyperedges": args.calib_min_total_hyperedges,
        "calib_max_total_hyperedges": args.calib_max_total_hyperedges,
        "repair_max_attempts": args.repair_max_attempts,
    }
    os.makedirs(final_save_path, exist_ok=True)

    print("\nPath configuration:")
    print(f"  Source (H5):  {h5_path}")
    print(f"  Destination:  {final_save_path}")

    print("\nHypergraph parameters:")
    for key, value in hypergraph_params.items():
        print(f"  {key:30s}: {value}")

    print("\n" + "="*80)
    print(" Starting H5 to hypergraph conversion...")
    print("="*80)

    convert_h5_directory(
        h5_path=h5_path,
        save_path=final_save_path,
        node_type_path=node_type_path,
        **hypergraph_params
    )

    print("\nAll hypergraph files have been generated.")

    print("\n" + "="*80)
    print(" Generating Visualizations (optional)...")
    print("="*80)

    generated_files = [f for f in os.listdir(final_save_path) if f.endswith('.pt')]

    if generated_files and args.visualize_count > 0:
        sample_files = generated_files[:args.visualize_count]
        print(f"\nVisualizing {len(sample_files)} sample files...")
        pbar_vis = tqdm(sample_files, desc="Visualizing")

        for file_to_visualize in pbar_vis:
            pbar_vis.set_description(f"Visualizing {file_to_visualize[:15]}")
            pt_file_path = os.path.join(final_save_path, file_to_visualize)
            visualize_hyperedges(pt_file_path, final_save_path, num_to_show=5)
    elif args.visualize_count > 0:
        print("No hypergraph files found to visualize.")

    print("\n" + "="*80)
    print(" Pipeline Completed Successfully!")
    print("="*80)
    print(f"\nOutput directory: {final_save_path}")
    print(f"Total files generated: {len(generated_files)}")
    print("\nGenerated files:")
    print("  - *.pt files          : Hypergraph data")
    print("  - *_hyperedges.png    : Visualization (sample)")
    print("\n" + "="*80)
