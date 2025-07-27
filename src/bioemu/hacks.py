import numpy as np
from scipy.optimize import minimize
from openmm import unit

def openmm_energy_and_gradient(x, simulation, eval_counter):
    # Reshape x to match number of atoms
    x_nm = x.reshape((-1, 3)) * unit.nanometer

    # Update positions in OpenMM
    simulation.context.setPositions(x_nm)

    # Compute energy and forces
    state = simulation.context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
    grad = -forces.reshape(-1)  # SciPy expects gradient (negative force)

    eval_counter[0] += 1  # Count evaluations
    return energy, grad

def minimize_with_scipy(simulation, maxiter=100000):
    # Get initial positions
    state = simulation.context.getState(getPositions=True)
    x0 = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer).reshape(-1)

    eval_counter = [0]

    result = minimize(
        fun=lambda x: openmm_energy_and_gradient(x, simulation, eval_counter),
        x0=x0,
        method="L-BFGS-B",
        jac=True,
        options={"gtol": 10.0, "maxiter": maxiter}  # OpenMM-like defaults
    )

    # Update OpenMM context to final positions
    x_final = result.x.reshape((-1, 3)) * unit.nanometer
    simulation.context.setPositions(x_final)

    return eval_counter[0]  # Return number of evaluations
    
from Bio.PDB import PDBParser
import numpy as np

def normalize_atom_name(atom_name):
    """Normalize atom names for matching (e.g., H -> H1)."""
    if atom_name == "H":  # BioEmu naming
        return "H1"
    return atom_name

def get_atom_list(pdb_file):
    """Return a list of (resSeq, resName, atomName) for each atom in PDB order."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_file)
    atoms = []
    for atom in structure.get_atoms():
        parent = atom.get_parent()
        resSeq = parent.get_id()[1]  # residue number
        resName = parent.get_resname().strip()
        atomName = normalize_atom_name(atom.get_name().strip())
        atoms.append((resSeq, resName, atomName))
    return atoms

def get_atom_set(pdb_file):
    """Return a set of (resSeq, resName, atomName) for each atom in PDB."""
    return set(get_atom_list(pdb_file))

def check_atom_match(ref_pdb, bioemu_pdb):
    ref_atoms = get_atom_set(ref_pdb)
    bioemu_atoms = get_atom_set(bioemu_pdb)

    if ref_atoms == bioemu_atoms:
        print("✅ Atom sets match exactly.")
        return True

    missing_in_bioemu = ref_atoms - bioemu_atoms
    extra_in_bioemu = bioemu_atoms - ref_atoms

    if missing_in_bioemu:
        print("❌ Missing atoms in BioEmu PDB:")
        for atom in sorted(missing_in_bioemu):
            print("  ", atom)

    if extra_in_bioemu:
        print("❌ Extra atoms in BioEmu PDB:")
        for atom in sorted(extra_in_bioemu):
            print("  ", atom)

    raise ValueError("Atom sets do NOT match between reference and BioEmu PDBs!")

def reorder_coordinates(ref_pdb, bioemu_pdb, bioemu_npy):
    """Reorder BioEmu coordinates to match reference PDB order."""
    ref_atoms = get_atom_list(ref_pdb)
    bioemu_atoms = get_atom_list(bioemu_pdb)

    mapping = []
    for ref_atom in ref_atoms:
        try:
            idx = bioemu_atoms.index(ref_atom)
            mapping.append(idx)
        except ValueError:
            raise ValueError(f"Atom {ref_atom} not found in BioEmu PDB (after normalization)!")

    mapping = np.array(mapping)

    print("bioemu_npy shape:", bioemu_npy.shape)
    print("mapping length:", mapping.shape)

    return bioemu_npy[:, mapping, :]
