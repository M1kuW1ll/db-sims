#!/usr/bin/env python3
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

_DATA_DIR = Path(__file__).parent / "data"

def load_gcp(
    latency_std_fraction: float = 0.15,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Load GCP region pairwise latency data

    Args:
        latency_std_fraction: std = fraction * mean for each entry

    Returns:
        region_names: list of GCP region IDs (length R)
        latency_mean: (R, R) matrix of mean latencies in seconds
        latency_std: (R, R) matrix of latency std deviations in seconds
    """
    latency_df = pd.read_csv(_DATA_DIR / "gcp_latency.csv")

    # Build ordered region list from the regions that appear in the latency data
    regions_in_data = sorted(
        set(latency_df["sending_region"]) | set(latency_df["receiving_region"])
    )
    region_index = {r: i for i, r in enumerate(regions_in_data)}
    n = len(regions_in_data)

    # Missing pairs get max observed latency as fallback
    max_ms = latency_df["milliseconds"].max()
    latency_ms = np.full((n, n), max_ms)
    np.fill_diagonal(latency_ms, 0.0)

    for _, row in latency_df.iterrows():
        i = region_index[row["sending_region"]]
        j = region_index[row["receiving_region"]]
        latency_ms[i, j] = row["milliseconds"]

    # use the mean of the two directions where both are available
    latency_ms = (latency_ms + latency_ms.T) / 2
    np.fill_diagonal(latency_ms, 1.0)  # 1ms to avoid log(0)

    latency_mean = latency_ms / 1000.0  # convert to seconds
    latency_std = latency_mean * latency_std_fraction
    latency_std = np.clip(latency_std, 1e-4, None)  # prevent zero std

    return regions_in_data, latency_mean, latency_std


def gcp_sources(
    region_names: List[str],
    source_regions: List[str],
    lambda_rate: float = 5.0,
    mu_val: float = 1.0,
    sigma_val: float = 0.5,
) -> List[tuple]:
    """
    Build a sources_config list for ExperimentConfig from named GCP regions

    Args:
        region_names: full ordered list returned by load_gcp()
        source_regions: GCP region IDs where sources are located
        lambda_rate: txs per second per source
        mu_val: lognormal mean of tx value
        sigma_val: lognormal std of tx value

    Returns:
        sources_config list of (name, region_idx, lambda_rate, mu_val, sigma_val)
    """
    index = {r: i for i, r in enumerate(region_names)}
    missing = [region for region in source_regions if region not in index]
    if missing:
        raise ValueError(f"Source regions not found in dataset: {missing}")
    return [
        (region, index[region], lambda_rate, mu_val, sigma_val)
        for region in source_regions
    ]


def subregion(
    region_names: List[str],
    latency_mean: np.ndarray,
    latency_std: np.ndarray,
    keep: List[str],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Slice the full GCP matrix down to a subset of regions.

    Args:
        region_names: full list from load_gcp()
        latency_mean: full (R, R) matrix
        latency_std: full (R, R) matrix
        keep: list of GCP region IDs to retain

    Returns:
        (subset_names, subset_latency_mean, subset_latency_std)
    """
    index = {r: i for i, r in enumerate(region_names)}
    missing = [region for region in keep if region not in index]
    if missing:
        raise ValueError(f"Regions not found in dataset: {missing}")

    idx = [index[region] for region in keep]
    return keep, latency_mean[np.ix_(idx, idx)], latency_std[np.ix_(idx, idx)]
