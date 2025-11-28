from __future__ import annotations

from typing import Tuple, Optional, Dict
import numpy as np

import argparse
#import json
import sys

from ase.io import read
from ase.build import surface, add_adsorbate, molecule, make_supercell
from ase.optimize import LBFGS
from ase.constraints import FixAtoms
from ase.atoms import Atoms

from fairchem.core import FAIRChemCalculator, pretrained_mlip


def adsorption_energy_h_from_poscar(
    poscar_path: str,
    miller: Tuple[int, int, int] = (1, 1, 1),
    layers: int = 4,
    vacuum: float = 12.0,
    repeat: Tuple[int, int, int] = (2, 2, 1),
    site_atom_index: Optional[int] = None,
    height: float = 2.0,
    relax: bool = True,
    fmax: float = 0.05,
    steps: int = 200,
    fix_n_bottom_layers: int = 1,
    reference: str = "H2",
    return_details: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute H adsorption energy on a catalyst surface using the eSEN OC25 model.

    Adsorption energy definition:
        E_ads = E(slab+H) - E(slab) - 1/2 * E(H2)    (if reference == "H2")
        E_ads = E(slab+H) - E(slab) - E(H)           (if reference == "H")

    Parameters
    ----------
    poscar_path : str
        Path to a VASP POSCAR (bulk) file.
    miller : (h, k, l)
        Miller indices for the surface cut.
    layers : int
        Number of atomic layers in the slab (before repeating).
    vacuum : float
        Vacuum thickness added along surface normal (+z).
    repeat : (nx, ny, nz)
        Supercell repeat of the slab (nz usually 1 for surfaces).
    site_atom_index : int or None
        Index (in the *final repeated slab*) of the surface atom on which to place H
        (H will be placed height Å above this atom along +z). If None, we choose the
        atom with the maximum z (top-most).
    height : float
        Height (Å) of the H adsorbate above the chosen surface atom, along +z.
    relax : bool
        If True, relax H2, clean slab, and adsorbed slab before energies are taken.
    fmax : float
        LBFGS force convergence threshold (eV/Å).
    steps : int
        Max LBFGS steps for each relaxation.
    fix_n_bottom_layers : int
        Number of bottom atomic layers to fix during slab relaxations.
    reference : {"H2", "H"}
        Reference state for hydrogen.
    return_details : bool
        If True, also return a dict with intermediate energies.

    Returns
    -------
    E_ads : float
        Adsorption energy in eV (negative => exothermic adsorption).
    details : dict
        Contains E_slab, E_H2 or E_H, E_slab_H, and metadata (if return_details=True).
    """
    # ---- set up the eSEN OC25 calculator ----
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p1")
    calc = FAIRChemCalculator(predictor, task_name="oc20")

    # ---- build slab from bulk POSCAR ----
    bulk: Atoms = read(poscar_path)  # bulk structure
    # Create a slab oriented by Miller indices with a given number of layers and vacuum.
    slab: Atoms = surface(bulk, miller, layers=layers, vacuum=vacuum, periodic=True)

    # Repeat (supercell) to reach desired lateral size.
    # `repeat` can be applied with make_supercell or slab.repeat; slab.repeat is simpler.
    slab = slab.repeat(repeat)

    # Ensure the surface normal is along +z (ASE's surface() does this by construction).
    # Determine which atoms belong to the bottom layers to fix them during relaxation.
    if fix_n_bottom_layers > 0:
        # Sort atoms by z to identify layers
        z_positions = slab.get_positions()[:, 2]
        # Use a simple geometric clustering by z with a tolerance
        tol = 0.3  # Å between layers (robust for many systems; adjust if needed)
        unique_layers = []
        for z in sorted(z_positions):
            if not unique_layers or abs(z - unique_layers[-1]) > tol:
                unique_layers.append(z)
        # Identify bottom layer z thresholds
        bottom_threshold = unique_layers[min(fix_n_bottom_layers - 1, len(unique_layers) - 1)]
        fixed_mask = z_positions <= bottom_threshold + 1e-6
        slab.set_constraint(FixAtoms(mask=fixed_mask))

    # Assign calculator for energy/forces
    slab.calc = calc

    # Optionally relax the clean slab
    if relax:
        opt = LBFGS(slab, logfile=None)
        opt.run(fmax=fmax, steps=steps)

    E_slab = float(slab.get_potential_energy())

    # ---- hydrogen reference energy ----
    if reference.upper() == "H2":
        H2 = molecule("H2")
        H2.calc = calc
        if relax:
            opt = LBFGS(H2, logfile=None)
            opt.run(fmax=fmax, steps=steps)
        E_ref = float(H2.get_potential_energy()) / 2.0
        ref_label = "E_H2_over_2"
    elif reference.upper() == "H":
        H = molecule("H")
        H.calc = calc
        if relax:
            opt = LBFGS(H, logfile=None)
            opt.run(fmax=fmax, steps=steps)
        E_ref = float(H.get_potential_energy())
        ref_label = "E_H"
    else:
        raise ValueError("reference must be 'H2' or 'H'")

    # ---- build adsorbed system (copy slab, add H at specified site) ----
    slab_H = slab.copy()

    # Choose adsorption site atom index
    if site_atom_index is None:
        # pick the top-most atom (max z)
        z = slab_H.get_positions()[:, 2]
        site_atom_index = int(np.argmax(z))
    else:
        if not (0 <= site_atom_index < len(slab_H)):
            raise IndexError(f"site_atom_index {site_atom_index} out of range for slab of size {len(slab_H)}")

    site_xy = slab_H[site_atom_index].position[:2]

    # Create an H atom and place it height Å above the chosen atom along +z.
    H_ads = molecule("H")
    # add_adsorbate places the adsorbate's COM at (x, y, z_top + height).
    # For a single H atom, COM ~ atomic position, which is what we want.
    add_adsorbate(slab_H, H_ads, height, position=tuple(site_xy))

    # Re-apply same calculator and constraints (fix bottom) to the adsorbed system.
    slab_H.calc = calc
    if fix_n_bottom_layers > 0:
        # Recreate the constraint mask on the copy
        z_positions = slab_H.get_positions()[:, 2]
        tol = 0.3
        unique_layers = []
        for z in sorted(z_positions[:len(slab)]):  # only consider original slab atoms for layers
            if not unique_layers or abs(z - unique_layers[-1]) > tol:
                unique_layers.append(z)
        bottom_threshold = unique_layers[min(fix_n_bottom_layers - 1, len(unique_layers) - 1)]
        fixed_mask = np.array([pos[2] <= bottom_threshold + 1e-6 for pos in slab_H.get_positions()])
        slab_H.set_constraint(FixAtoms(mask=fixed_mask))

    # Relax the adsorbed system if requested
    if relax:
        opt = LBFGS(slab_H, logfile=None)
        opt.run(fmax=fmax, steps=steps)

    E_slab_H = float(slab_H.get_potential_energy())

    # ---- adsorption energy ----
    E_ads = E_slab_H - E_slab - E_ref

    details = {
        "E_slab": E_slab,
        "E_slab_plus_H": E_slab_H,
        "ref_label": E_ref,
        "E_adsorption": E_ads,
        "site_atom_index": int(site_atom_index),
        "miller": tuple(miller),
        "layers": int(layers),
        "vacuum": float(vacuum),
        "repeat": tuple(repeat),
        "height": float(height),
        "reference": reference.upper(),
        "relaxed": bool(relax),
        "fmax": float(fmax),
        "steps": int(steps),
        "fix_n_bottom_layers": int(fix_n_bottom_layers),
    }

    return (E_ads, details) if return_details else (E_ads, {})


# ---------------------------
# Minimal example of usage:
# ---------------------------
# E_ads, info = adsorption_energy_h_from_poscar(
#     poscar_path="POSCAR",           # bulk POSCAR path
#     miller=(1, 0, 0),               # choose surface orientation
#     layers=6,                       # number of atomic layers
#     vacuum=12.0,                    # vacuum thickness (Å)
#     repeat=(3, 3, 1),               # lateral supercell
#     site_atom_index=None,           # pick top-most atom automatically, or give an index
#     height=2.0,                     # 2 Å above chosen surface atom
#     relax=True,                     # relax all configs with LBFGS
#     fmax=0.05,
#     steps=200,
#     fix_n_bottom_layers=2,          # fix bottom 2 layers
#     reference="H2",                 # use 1/2 E(H2) reference
#     return_details=True,
# )
# print("E_ads (eV):", E_ads)
# print(info)

# --- Main execution block for command line interface (CLI) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute H adsorption energy using eSEN OC25 model.")
    # Required positional argument
    parser.add_argument("poscar_path", type=str, help="Path to a VASP POSCAR (bulk) file.")
    # Optional arguments with defaults others may be added
    parser.add_argument("--miller", type=int, nargs=3, default=[1, 1, 1], help="Miller indices (h k l) for the surface cut. Default: 1 1 1.")
    parser.add_argument("--layers", type=int, default=4, help="Number of atomic layers in the slab. Default: 4.")
    parser.add_argument("--relax", type=str,  default="False",  choices=["True", "False"], help="If True, relax structures before energies are taken. Default: True.")
    parser.add_argument( "--reference",  type=str,  default="H2",  choices=["H2", "H"],  help="Reference state for hydrogen ('H2' or 'H'). Default: H2.")

    args = parser.parse_args()

    # Convert the 'relax' string argument to a boolean
    relax_bool = args.relax.lower() == 'true'

    try:
        # Call the main function with parsed arguments
        E_ads, details = adsorption_energy_h_from_poscar(
            poscar_path=args.poscar_path,
            miller=tuple(args.miller),
            layers=args.layers,
            relax=relax_bool,
            reference=args.reference,
            return_details=True # Always return details for CLI output
        )

        # Output the result as a single JSON object to stdout
        # This makes it easy for the calling process to parse.
        output = {
            "E_adsorption": E_ads,
            "details": details
        }
        print(E_ads)

    except Exception as e:
        # Print errors to stderr and exit with a non-zero code
        #print(json.dumps({"error": str(e), "filepath": args.poscar_path}), file=sys.stderr)
        
        sys.exit(1)

