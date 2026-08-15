#!/usr/bin/env python3
"""
root_to_lhco_csv.py
====================

Convert a Delphes-style .root file into an LHCO-like flat .csv file.

Usage
-----
    python root_to_lhco_csv.py input.root output.csv
    python root_to_lhco_csv.py input.root output.csv --tree Delphes
    python root_to_lhco_csv.py input.root output.csv --list-branches   # just inspect the file

Requirements
------------
    pip install uproot awkward pandas numpy

What it assumes
----------------
This targets the common Delphes truth-level layout:

    Particle.PID, Particle.Eta, Particle.Phi, Particle.PT,
    Particle.Mass, Particle.Charge          (one row per generator particle)
    MissingET.MET, MissingET.Eta, MissingET.Phi   (one value per event)

If your file uses different branch names, run with --list-branches
first to see what's actually inside, then pass --pid-branch,
--eta-branch, etc. (see `python root_to_lhco_csv.py --help`) or edit
CANDIDATES below.

Output format
--------------
One row per physics object (particle or MET), tagged with which event
and which slot within the event it belongs to:

    event, obj, typ, eta, phi, pt, jmass, ntrk, btag, hadem

    typ: 0=photon 1=electron 2=muon 3=tau 4=jet 6=MET
    ntrk: track count for jets, but here reused as *charge* for
          leptons/quarks, matching your original code's convention.
    btag: 1 if the object is a b-quark/b-jet, else 0.
"""

import argparse
import sys

import numpy as np
import awkward as ak
import pandas as pd
import uproot


# ---------------------------------------------------------------------
# Candidate branch names, tried in order, per logical field.
# Extend this if your file uses a naming scheme not listed here.
# ---------------------------------------------------------------------
CANDIDATES = {
    "pid":    ["Particle/Particle.PID", "GenParticle.PID"],
    "eta":    ["Particle/Particle.Eta", "GenParticle.Eta"],
    "phi":    ["Particle/Particle.Phi", "GenParticle.Phi"],
    "pt":     ["Particle/Particle.PT", "GenParticle.PT"],
    "mass":   ["Particle/Particle.Mass", "GenParticle.Mass"],
    "charge": ["Particle/Particle.Charge", "GenParticle.Charge"],
    "met":     ["MissingET/MissingET.MET", "GenMissingET.MET"],
    "met_eta": ["MissingET/MissingET.Eta", "GenMissingET.Eta"],
    "met_phi": ["MissingET/MissingET.Phi", "GenMissingET.Phi"],
}

PID_TO_TYP = {
    22: (0, False), # Photon
    11: (1, False), # Electron
    13: (2, False), # Muon
    15: (3, False), # Tau
    1: (4, False), 2: (4, False), 3: (4, False), 4: (4, False), # Quarks d, u, s ,c to Jets
    5: (4, True), # Quark b to jet
    6: (4, False), 
    7: (4, False), # Quarks t, b' & t' to jets
    8: (4, False), 
}


def open_tree(root_path, tree_name=None):
    """
    Open the ROOT file and return the tree to read from.

    Delphes files normally contain a single TTree called 'Delphes'.
    If a different/unknown tree name is used, we fall back to picking
    the first TTree found in the file, and tell the user what we chose.
    """
    f = uproot.open(root_path)

    if tree_name is not None:
        if tree_name not in f:
            raise KeyError(
                f"Tree '{tree_name}' not found. Top-level keys in file: {f.keys()}"
            )
        return f[tree_name]

    # look for a TTree, preferring the conventional Delphes name
    ttrees = [k for k, cls in zip(f.keys(), f.classnames().values())] \
        if hasattr(f, "classnames") else f.keys()

    if "Delphes" in f or "Delphes;1" in f:
        print("Auto-detected tree: 'Delphes'")
        return f["Delphes"]

    # fall back: first object that behaves like a TTree
    for key in f.keys():
        obj = f[key]
        if isinstance(obj, uproot.behaviors.TTree.TTree):
            print(f"Auto-detected tree: '{key}' (no 'Delphes' tree found)")
            return obj

    raise RuntimeError(f"No TTree found in file. Top-level keys: {f.keys()}")


def resolve_branch(tree, field_name, override=None):
    """
    Find the actual branch name in `tree` for a logical field
    (e.g. 'pid'), trying the user override first, then CANDIDATES.
    """
    keys = set(tree.keys())

    if override is not None:
        if override not in keys:
            raise KeyError(
                f"--{field_name.replace('_', '-')}-branch '{override}' not found in tree. "
                f"Available branches: {sorted(keys)}"
            )
        return override

    for candidate in CANDIDATES[field_name]:
        if candidate in keys:
            return candidate

    raise KeyError(
        f"Could not auto-detect a branch for '{field_name}'. Tried: "
        f"{CANDIDATES[field_name]}. Available branches in this tree:\n"
        f"{sorted(keys)}\n"
        f"Pass --{field_name.replace('_', '-')}-branch <name> to specify it manually."
    )


def load_arrays(root_path, tree_name=None, overrides=None):
    """
    Open the file, resolve every branch, and return them as
    awkward Arrays (jagged for particles, flat for MET).
    """
    overrides = overrides or {}
    tree = open_tree(root_path, tree_name)

    resolved = {
        field: resolve_branch(tree, field, overrides.get(field))
        for field in CANDIDATES
    }
    print("Resolved branches:")
    for field, branch in resolved.items():
        print(f"  {field:9s} -> {branch}")

    data = tree.arrays(list(resolved.values()), library="ak")

    pid    = data[resolved["pid"]]
    eta    = data[resolved["eta"]]
    phi    = data[resolved["phi"]]
    pt     = data[resolved["pt"]]
    mass   = data[resolved["mass"]]
    charge = data[resolved["charge"]]

    # MissingET is stored as a length-1 jagged array per event in
    # Delphes (a "collection" with exactly one entry) -- flatten that
    # inner singleton dimension down to one scalar per event.
    met      = ak.firsts(data[resolved["met"]])
    met_eta  = ak.firsts(data[resolved["met_eta"]])
    met_phi  = ak.firsts(data[resolved["met_phi"]])

    return pid, eta, phi, pt, mass, charge, met, met_eta, met_phi


def classify_and_build(pid, eta, phi, pt, mass, charge, met, met_eta, met_phi):
    """
    Vectorized PID classification + MET-appending. Returns one
    awkward Array per output column, still jagged (event x objects).
    """
    n_events = len(pid)
    ap = np.abs(pid)

    is_photon   = ap == 22
    is_electron = ap == 11
    is_muon     = ap == 13
    is_tau      = ap == 15
    is_quark    = (ap >= 1) & (ap <= 8)
    is_known    = is_photon | is_electron | is_muon | is_tau | is_quark

    typ = ak.where(is_photon, 0,
          ak.where(is_electron, 1,
          ak.where(is_muon, 2,
          ak.where(is_tau, 3,
          ak.where(is_quark, 4, -1)))))
    btag = ak.where(is_quark & (ap == 5), 1, 0)
    hadem = ak.zeros_like(typ)

    mask = is_known
    typ_f, eta_f, phi_f, pt_f = typ[mask], eta[mask], phi[mask], pt[mask]
    jmass_f, ntrk_f, btag_f, hadem_f = mass[mask], charge[mask], btag[mask], hadem[mask]

    def col(flat_values):
        return ak.Array(np.asarray(flat_values))[:, np.newaxis]

    typ_out   = ak.concatenate([typ_f,   col(np.full(n_events, 6))], axis=1)
    eta_out   = ak.concatenate([eta_f,   col(met_eta)], axis=1)
    phi_out   = ak.concatenate([phi_f,   col(met_phi)], axis=1)
    pt_out    = ak.concatenate([pt_f,    col(met)], axis=1)
    jmass_out = ak.concatenate([jmass_f, col(np.zeros(n_events))], axis=1)
    ntrk_out  = ak.concatenate([ntrk_f,  col(np.zeros(n_events))], axis=1)
    btag_out  = ak.concatenate([btag_f,  col(np.zeros(n_events))], axis=1)
    hadem_out = ak.concatenate([hadem_f, col(np.zeros(n_events))], axis=1)

    return typ_out, eta_out, phi_out, pt_out, jmass_out, ntrk_out, btag_out, hadem_out


def to_flat_dataframe(typ, eta, phi, pt, jmass, ntrk, btag, hadem):
    """
    Flatten the jagged (event x object) arrays into a single tidy
    table with explicit 'event' and 'obj' index columns, ready for
    pandas.to_csv().
    """
    n_events = len(typ)

    # event index, broadcast to match each event's object count,
    # then flattened -- this is what stamps every row with which
    # event it came from.
    event_idx = ak.broadcast_arrays(ak.local_index(typ, axis=0), typ)[0]
    obj_idx = ak.local_index(typ, axis=1)  # 0..n_objects-1 within each event

    df = pd.DataFrame({
        "event": ak.flatten(event_idx).to_numpy(),
        "obj":   ak.flatten(obj_idx).to_numpy(),
        "typ":   ak.flatten(typ).to_numpy(),
        "eta":   ak.flatten(eta).to_numpy(),
        "phi":   ak.flatten(phi).to_numpy(),
        "pt":    ak.flatten(pt).to_numpy(),
        "jmass": ak.flatten(jmass).to_numpy(),
        "ntrk":  ak.flatten(ntrk).to_numpy(),
        "btag":  ak.flatten(btag).to_numpy(),
        "hadem": ak.flatten(hadem).to_numpy(),
    })
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_root", help="path to input .root file")
    parser.add_argument("output_csv", nargs="?", help="path to output .csv file")
    parser.add_argument("--tree", default=None, help="TTree name (default: auto-detect)")
    parser.add_argument("--list-branches", action="store_true",
                         help="just print available branches in the tree and exit")
    for field in CANDIDATES:
        parser.add_argument(f"--{field.replace('_', '-')}-branch", default=None,
                             help=f"override auto-detected branch name for '{field}'")
    args = parser.parse_args()

    if args.list_branches:
        tree = open_tree(args.input_root, args.tree)
        print(f"\nBranches in tree (n_events={tree.num_entries}):")
        for k in tree.keys():
            print(f"  {k}")
        return

    if not args.output_csv:
        parser.error("output_csv is required unless --list-branches is given")

    overrides = {field: getattr(args, f"{field}_branch") for field in CANDIDATES}

    print(f"Reading '{args.input_root}' ...")
    arrays = load_arrays(args.input_root, args.tree, overrides)

    print("Classifying particles and appending MET ...")
    cols = classify_and_build(*arrays)

    print("Flattening to a tidy table ...")
    df = to_flat_dataframe(*cols)

    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(df)} rows across {df['event'].nunique()} events to '{args.output_csv}'")


if __name__ == "__main__":
    main()