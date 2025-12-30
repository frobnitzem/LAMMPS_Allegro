import numpy as np

def abc_ang_from_L(L):
    abc = (L**2).sum(1)**0.5
    ang = [np.arccos(np.dot(L[i],L[j])/(abc[i]*abc[j])) \
            for i,j in [(1,2),(0,2),(0,1)]]
    return abc, np.array(ang)

def write_pdb(out, L, atoms):
    abc, ang = abc_ang_from_L(L)
    ang *= 180./np.pi
    out.write("CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1\n"%(\
                                abc[0],abc[1],abc[2],ang[0],ang[1],ang[2]))
    fmt = "ATOM  %5d%s%8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n"

    for a in atoms:
        rname, chain, rnum = a[1]
        name = " %4s %3s %c%4d    "%(a[2], rname, chain, rnum)
        out.write(fmt%(a[0], name, a[4], a[5], a[6], 1, 1, a[2]))

def from_mass(m: float) -> str:
    ref_masses = [
        (1.0, "H"),
        (12.0, "C"),
        (14.0, "N"),
        (16.0, "O"),
        (32.0, "S"),
    ]
    for r, c in ref_masses:
        if abs(r-m) < 0.5:
            return c
    raise KeyError(f"Unknown mass: {m}")

def parse_lammps_data(fname: str):
    atoms = []
    # Atom-ID Molecule-ID Atom-type Charge X Y Z
    insec = None
    L = np.eye(3)*100
    types = {}
    with open(fname, "r") as inp:
        for line in inp:
            tok = line.split("#", 1)[0].split()
            if len(tok) == 0:
                continue
            elif len(tok) == 1:
                if insec == "Atoms":
                    break
                insec = tok[0]
                continue
            elif insec is None:
                if tok[-1] == "xhi":
                    L[0,0] = float(tok[1])-float(tok[0])
                elif tok[-1] == "yhi":
                    L[1,1] = float(tok[1])-float(tok[0])
                elif tok[-1] == "zhi":
                    L[2,2] = float(tok[1])-float(tok[0])
            elif insec == "Masses":
                types[tok[0]] = from_mass( float(tok[1]) )
            elif insec == "Atoms" and len(tok) >= 7:
                c = types[ tok[2] ]
                atoms.append((int(tok[0]), int(tok[1]), c, float(tok[3]),
                              float(tok[4]), float(tok[5]), float(tok[6])
                             ))

    atoms.sort()
    return L, atoms

def name_res_chain(atoms):
    # This would be the correct way to do it, if the
    # molecule numbers changed for each water...
    mol = [a[1] for a in atoms]
    counts = np.bincount(mol)

    cnum = ord('A') # name protein chains sequentially from 'A'
    wnum = 0 # number waters sequentially from 1

    chains = {}
    for i,n in enumerate(counts):
        if n == 0:
            continue
        elif n == 3:
            wnum += 1
            chains[i] = ('SOL', ' ', wnum)
        else:
            chains[i] = ('UNK', chr(cnum), 1)
            cnum += 1
    for i, a in enumerate(atoms):
        atoms[i] = (a[0], chains[a[1]]) + a[2:]

def renumber_waters(atoms):
    # Let each new "O" in molecule 2 be renumbered
    # to a successive molecule ID.
    n = 1
    for i, a in enumerate(atoms):
        if a[1] == 2:
            if a[2] == 'O':
                n += 1
            atoms[i] = (a[0], n) + a[2:]

def main(argv):
    assert len(argv) == 3, f"Usage: <in.data> <out.pdb>"

    L, atoms = parse_lammps_data(argv[1])
    renumber_waters(atoms)
    name_res_chain(atoms)

    with open(argv[2], "w") as out:
        write_pdb(out, L, atoms)

if __name__=="__main__":
    import sys
    main(sys.argv)

