import pandas as pd

# Redox potential calculation method for Born-Haber thermodynamic cycle
# Works best on A + H2 -> B or A -> B reactions because non-ideal entropic effects (roughly) cancel out between solvated and gas phases
def redox_potential_BornHaber(
    unreducedDF,
    reducedDF,
    H2DF,
    # num. e-
    nReduction: int = 2,
    # J/C (i.e., V)
    referenceState: float = 0.0,
):
    # Ensure PANDAS dataframe inputs
    assert (
        isinstance(unreducedDF, pd.DataFrame)
        and isinstance(reducedDF, pd.DataFrame)
        and isinstance(H2DF, pd.DataFrame)
    )

    # Inputs

    # Name of variable
    # Identifying column of variable
    # Units of variable

    # System Temperature
    # "Temperature (K)"
    # K
    (unredTemp, redTemp, H2Temp) = (
        unreducedDF.loc[0, "Temperature (K)"].item(),
        reducedDF.loc[0, "Temperature (K)"].item(),
        H2DF.loc[0, "Temperature (K)"].item(),
    )
    tempTuple = (unredTemp, redTemp, H2Temp)

    # Gas-phase Electronic Structure Energy
    # "System Energy (Hartree)"
    # Hartree
    (unredGESE, redGESE, H2GESE) = (
        unreducedDF.loc[0, "System Energy (Hartree)"].item(),
        reducedDF.loc[0, "System Energy (Hartree)"].item(),
        H2DF.loc[0, "System Energy (Hartree)"].item(),
    )
    GESETuple = (unredGESE, redGESE, H2GESE)

    # Thermal Correction to Gas-phase Electronic Structure Energy
    # "Thermal correction to Energy (kcal/mol)"
    # (kcal/mol)
    (unredGESETherm, redGESETherm, H2GESETherm) = (
        unreducedDF.loc[0, "Thermal correction to Energy (kcal/mol)"].item(),
        reducedDF.loc[0, "Thermal correction to Energy (kcal/mol)"].item(),
        H2DF.loc[0, "Thermal correction to Energy (kcal/mol)"].item(),
    )
    GESEThermTuple = (unredGESETherm, redGESETherm, H2GESETherm)

    # Thermal Correction to Enthalpy
    # "Thermal correction to Enthalpy (kcal/mol)"
    # kcal/mol
    (unredThermEnthalpy, redThermEnthalpy, H2ThermEnthalpy) = (
        unreducedDF.loc[0, "Thermal correction to Enthalpy (kcal/mol)"].item(),
        reducedDF.loc[0, "Thermal correction to Enthalpy (kcal/mol)"].item(),
        H2DF.loc[0, "Thermal correction to Enthalpy (kcal/mol)"].item(),
    )
    thermEnthalpyTuple = (unredThermEnthalpy, redThermEnthalpy, H2ThermEnthalpy)

    # Total Entropy
    # "Total Entropy (cal/mol-K)"
    # cal/mol*K
    (unredEntropy, redEntropy, H2Entropy) = (
        unreducedDF.loc[0, "Total Entropy (cal/mol-K)"].item(),
        reducedDF.loc[0, "Total Entropy (cal/mol-K)"].item(),
        H2DF.loc[0, "Total Entropy (cal/mol-K)"].item(),
    )
    entropyTuple = (unredEntropy, redEntropy, H2Entropy)

    # Solvated System Energy
    # COSMO energy (Hartree)
    # Hartree
    (unredSolvSysEnergy, redSolvSysEnergy, H2SolvSysEnergy) = (
        unreducedDF.loc[0, "Solvated System Energy (Hartree)"].item(),
        reducedDF.loc[0, "Solvated System Energy (Hartree)"].item(),
        H2DF.loc[0, "Solvated System Energy (Hartree)"].item(),
    )
    solvSysEnergyTuple = (unredSolvSysEnergy, redSolvSysEnergy, H2SolvSysEnergy)

    # Calculations
    # Thermal Correction to Gibbs Free Energy Gas Phase
    gibbsGasThermTuple = []
    # Thermal-corrected Gibbs Free Energy in Gas Phase
    gibbsGasTuple = []
    # Thermal-corrected Gibbs Free Energy in Solution
    gibbsSolTuple = []

    # kcal/mol
    for temp, sysEnergy, thermSysEnergy, thermEnthalpy, entropy, solvSysEnergy in zip(
        tempTuple,
        GESETuple,
        GESEThermTuple,
        thermEnthalpyTuple,
        entropyTuple,
        solvSysEnergyTuple,
    ):
        # Thermally corrected gas phase system energy of reactant or product
        # kcal/mol
        corrSysEnergy = sysEnergy * 627.503 + thermSysEnergy

        # Thermal correction to Gibbs free energy of reactant or product in gas phase
        # kcal/mol
        gibbsGasTherm = thermEnthalpy - (temp * entropy / 1000)

        # Gibbs free energy of reactant or product in the gas phase
        # kcal/mol
        # We use the system energy in gase phase BEFORE thermal corrections at this step
        # This is because the thermal correction to Gibbs free energy (gibbsGasTherm) already corrects the system energy
        gibbsGas = sysEnergy + gibbsGasTherm

        # Gibbs free energy of reactant or product in solution
        # kcal/mol
        # We use the thermally corrected gas-phase system energy in this step
        # This is because the solvated system energy was determined with thermal corrections in mind
        # So, the energy of solvation needs to be in reference to two thermally corrected systems.
        gibbsSol = gibbsGas + (solvSysEnergy * 627.503 - corrSysEnergy)

        gibbsGasThermTuple.append(gibbsGasTherm)
        gibbsGasTuple.append(gibbsGas)
        gibbsSolTuple.append(gibbsSol)

    # Gibbs free energy
    # Reduced form - unreduced form - H2
    # NOTE:
    # On the PCET reaction assumption that # of electrons equals number of hydrogens,
    # 2 electrons means we can use 2 protons (or 1 unit of Gibbs free energy from simulated H2)
    # 1 electron means we can use 1 proton (or 1/2 unit of Gibbs free energy from simulated H2)
    # So, the general stoichiometric scaling is '# electrons'/2 or nReduction/2
    # kcal/mol
    gibbsRxn = gibbsSolTuple[1] - gibbsSolTuple[0] - (nReduction/2)*gibbsSolTuple[2]

    # J/mol
    gibbsRxn = gibbsRxn * 4184

    # C/mol e-
    faradayConst = 96485.33212331

    # J/C (i.e., V)
    redoxPotential = gibbsRxn / (-1 * nReduction * faradayConst)

    if referenceState is not None:
        # J/C (i.e., V)
        redoxPotential = redoxPotential - referenceState

    # J/c (i.e., V)
    return redoxPotential
