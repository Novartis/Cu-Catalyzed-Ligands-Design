# Ligands_Design

To further improve the efficiency of the hydroxylation reaction, it is essential to design and screen a large number of different ligand structures and explore the optimal reaction conditions. Traditional methods necessitate conducting numerous experiments, which are both time-consuming and resource-intensive. In our paper, we introduce machine learning methods to help us to increase our efficiency in ligands optimization and design.

## Getting started

To get ready for the ligands calculation you probably need to have rdkit and xtb installed on your machine. Rdkit is package for cheminformatics related tasks https://www.rdkit.org/docs/GettingStartedInPython.html, and xtb is package prvoding semiempirical quantum mechanical methods GFNn-xTB,https://xtb-docs.readthedocs.io/en/latest/setup.html. To perform those calcuations on the proposed ligands,in our case, all the calculations are being done in Novartis high performance computing servers with some internal built packages.

## Modify the ligands

Go for **Ligands_AddMetal_Steric_Desriptors__1_.ipynb**, the code includes manipulating the ligands molecules and adding Copper attaching to the ligands atoms, and how to generate steric properties after the xtb calculations are being done. 
1st Step:

Generating Copper complex from the ligands, creating initial sdf files for further xtb  calculations,
Once the ligands names and related sdf files are ready,you can generate optimized ligands' sdf files, an example is provided as below:

for i in names:
    path_1 = f"{home}/SIOC/Molecular_Generation/SIOC_Ligands/masm_conf_{i}.sdf"
    path_2 = f"{home}/SIOC/Molecular_Generation/SIOC_Ligands/masm_conf_xtb_{i}"
    path_3 = f"{home}/SIOC/Molecular_Generation/SIOC_Ligands/masm_conf_xtb_{i}.out"
    path_4 = f"{home}/SIOC/Molecular_Generation/SIOC_Ligands/masm_conf_xtb_{i}.error"
    !xtb {path_1} --opt --namespace {path_2} > {path_3} 2> {path_4}

The optimized ligands sdf files could be found under ligands folder. 
For further quantum propertis geneartions, we used internal used package for this purpose and in this jupyternotebook, exspecially molpipe which is for generating high-quality 3D configurations of compounds and easy quantum chemistry property calculations.

example code shared as below for generating properties using xtb

    def get_all_properties_xtb(molobj,solvent="h2o",xtb_options={}):
    <!-- """
    Get all the properties Rainer asked for

    Molecular properties:
        - Relative energy over minimum in water and chloroform, or Boltzmann
          weights, or both
        - Total energy as printed in output
        - Electronic energy (printed out in JSON) (SCC energy)
        - G(solv) @H2O and @Chloroform (xtb output)
        - HOMO-LUMO gap (@ALPB-H2O and @ALPB-CHCl3, requires just a single
          point @chloroform, no need to optimize geo)
        - Dipole moment (total, in Debye)
        - Orbital energies from HOMO-2 to LUMO+2 = 6 entries
        - Global electrophilicity index @H2O and @Chloroform (flag --vomega;
          needs a separate single point run on final geometry;  xtb <input>
          --gfn 2 --alpb h2o OR –alpb chcl3 –-vomega)

        - sasa

    Atomic properties
        - Partial charges
        ? Orbital coefficients on atoms for HOMO-2 to LUMO+2
        - Fukui indices   f(+), f(-), and f(0) ; @H2O and CHCl3 ; flag
          --vfukui; needs a separate single point run on final geometry; xtb
          <input> --gfn 2 --alpb h2o OR –alpb chcl3 –-vfukui

    Bond properties:
        - Bond lengths (all that are printed or just those without H, Peter?)
        - Bond orders (same? Or do we include X-H?)

    for each conformer

    """ -->

    _logger.info(f"Calculating all XTB descriptors for {solvent}")

    # xtb calculation options
    single_point = {
        "gfn": 2,
        "alpb": solvent,
    }

    electrophilicity = {
        "gfn": 2,
        "alpb": solvent,
        "vomega": None,
    }

    fukui = {
        "gfn": 2,
        "alpb": solvent,
        "vfukui": None,
    }

    # init calculator
    calc = xtb.XtbCalculator(**xtb_options)

    # start calculating
    results_sp = calc.calculate(molobj, single_point)

    # add weights to results_sp
    # Calculate Boltzmann probability weights
    energies = [properties[xtb.COLUMN_ENERGY] for properties in results_sp]
    energies = np.array(energies)
    energies *= units.hartree_to_kcalmol
    weights = chembridge.get_boltzmann_weights(energies)

    for i, weight in enumerate(weights):
        results_sp[i]["boltzmann_weight"] = weight

    # Get the fukui and omega properties
    results_fukui = calc.calculate(molobj, fukui)
    results_omega = calc.calculate(molobj, electrophilicity)

    properties = []

    for result_sp, result_fukui, result_omega in zip(results_sp, results_fukui, results_omega):
        result = {**result_sp, **result_fukui, **result_omega}
        result = {f"{solvent}_{key}": val for key, val in result.items()}
        properties.append(result)

    return properties__


With the optimized ligands sdf file, you could further using Morfeus package https://digital-chemistry-laboratory.github.io/morfeus/notes.html, for further steric descriptors generation.


## Anaylsis & Prediction

Go for **Ligands_Analysis_Prediction.ipynb** file for detailed models on training and prediction. The databased training_set.csv and Unsynthesized_ligands_descriptors.csv used in the jupyternotebook could be found under data folder.
=======
# Cu-Catalyzed-Ligands-Design
By establishing machine learning (ML) models, the design of ligands and optimization of reaction conditions were effectively facilitated
>>>>>>> origin/main
