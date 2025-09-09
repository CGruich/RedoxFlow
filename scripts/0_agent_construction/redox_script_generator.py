import os
import itertools
import directives as dirs
from typing import Union

# We have a lot of classes that represent directives.
# Here, we make a class that takes the directives and sequentially combines them into an input script.
# This is a base class that can be formulated into classes that prepare custom input scripts (e.g., redox potential)

# Purpose: Provides a baseline, boiler-plate class from which to build off of for custom NWChem input script generation
# Author: Cameron Gruich


# directiveList: A list of NWChem input script directives derived from the NWChemDirective() class
#                For example, "dft" or "scratch_dir" in NWChem can be defined by DFTDirective(NWChemDirective) and ScratchDirDirective(NWChemDirective) classes
# Returns self.inputStr str: The entire NWChem input script returned as a multi-line string variable.
class NWChemInputConstructor(object):
    def __init__(self, directiveList: list = None):
        if directiveList is not None:
            for directive in directiveList:
                assert isinstance(directive, dirs.NWChemDirective)
        self.directiveList = directiveList
        self.inputStr = "EMPTY"

        # For combinatorial batch generation of input scripts,
        # This is a dict
        self.inputs = {}
        # This is the parent folder that we save all the unique input scripts for,
        self.inputScriptFolderTemplate = None
        # This is a switch that activates some batch-mode functionality for script generation. None by default
        self.batch = None

    def __str__(self):
        self.construct_input_script()
        return self.inputStr

    def construct_input_script(self):
        if self.directiveList is not None:
            self.inputStr = ""
            for directiveInd in range(len(self.directiveList)):
                directiveStr = self.directiveList[directiveInd].directiveStr
                assert "None" not in directiveStr
                if directiveInd != (len(self.directiveList) - 1):
                    addStr = f"{directiveStr}\n"
                else:
                    addStr = f"{directiveStr}"
                self.inputStr = self.inputStr + addStr
            return self.inputStr
        else:
            self.inputStr = "EMPTY"
            return self.inputStr

    # If batch-mode is activated, this function generates all possible combinations of input scripts from the specified variables.
    # Generated input combinations are saved in self.inputScriptSpace, which represents the entire batch of adjustable DFT options specified.
    # NOTICE: self.inputScriptSpace is not the collection of input scripts; rather, it is all the combinations of settings we want to vary, which we can use to make input scripts.
    # Returns self.inputScriptSpace list: All possible input scripts options combinatorially from the specified options, stored in a list.
    def batch_combinations(self):
        assert type(self.inputs) is dict
        assert len(self.inputs) != 0

        combinatoricInput = []

        for key, value in self.inputs.items():
            if type(value) is list:
                combinatoricInput.append(value)
            else:
                combinatoricInput.append([value])
        # Cartesian product
        cartProduct = itertools.product(*combinatoricInput)
        # Convert into a list of lists where each inner list represents a unique input script
        self.inputScriptSpace = [list(inputScript) for inputScript in cartProduct]

    # Save the generated input scripts.
    # This generates a unique folder for every input script and saved to outputFolderParent

    # outputFolderParent str: Where to save the input script folders
    def save_to_file(self, outputFolderParent: str = "."):
        assert self.batch is not None

        if not self.batch:
            if self.inputScriptFolderTemplate is None:
                outputFolder = os.path.join(outputFolderParent, "inputScript")
                os.makedirs(outputFolder, exist_ok=True)
                script = self.construct_input_script()
                with open(
                    os.path.join(outputFolder, "inputScript.nw"), "w"
                ) as inputScriptObj:
                    inputScriptObj.write(script)
            else:
                outputFolder = os.path.join(
                    outputFolderParent, self.inputScriptFolderTemplate
                )
                os.makedirs(outputFolder, exist_ok=True)
                scriptName = f"{self.inputScriptFolderTemplate}.nw"
                script = self.construct_input_script()
                with open(
                    os.path.join(outputFolder, scriptName), "w"
                ) as inputScriptObj:
                    inputScriptObj.write(script)
                return script

        else:
            if self.inputScriptFolderTemplate is None:
                for scriptInd in range(len(self.scripts)):
                    scriptIndPlus1 = scriptInd + 1
                    inputScriptBaseName = "inputScript" + str(scriptIndPlus1)
                    outputFolder = os.path.join(outputFolderParent, inputScriptBaseName)
                    outputFile = inputScriptBaseName + ".nw"
                    os.makedirs(outputFolder, exist_ok=True)
                    with open(
                        os.path.join(outputFolder, outputFile), "w"
                    ) as inputScriptObj:
                        inputScriptObj.write(self.scripts[scriptInd])
                return self.scripts
            else:
                for scriptInd in range(len(self.scripts)):
                    scriptIndPlus1 = scriptInd + 1
                    outputFolder = os.path.join(
                        outputFolderParent,
                        self.inputScriptFolderTemplate + str(scriptIndPlus1),
                    )
                    os.makedirs(outputFolder, exist_ok=True)
                    outputFile = (
                        self.inputScriptFolderTemplate + str(scriptIndPlus1) + ".nw"
                    )
                    with open(
                        os.path.join(outputFolder, outputFile), "w"
                    ) as inputScriptObj:
                        inputScriptObj.write(self.scripts[scriptInd])
                return self.scripts

    # Compatibility shim for callers that expect write_input()
    def write_input(self, outputFolderParent: str = "."):
        return self.save_to_file(outputFolderParent=outputFolderParent)


# ----------------------------

# ----------------------------
# Now, let's make a class that constructs scripts for redox potential calculation.

# Purpose: Generates NWChem redox potential input scripts, with built-in flexibility to define other calculations or perform simultaneous benchmarking.
#          This class is able to generate single input scripts OR generate scripts in batches.
#          To enable batch-mode, input variables to the class should be given as a list.
#          (e.g., xc = ["option1", "option2"] will generate two separate input scripts with two different exchange-correlation functionals)
#          Multiple lists can be provided, and this class will generate input scripts out of all possible combinations of values specified.
#          (e.g., xc = ["option1", "option2"] and geometry = ["geometry1", "geometry2"] produces 4 unique input script combinations)
#          If some setting is considered fixed and constant, it does not need to be specified as a list, (e.g., scratch_dir="/constant/directory")
# Author: Cameron Gruich


# titleGeomOptimizer str or list: Title of gas-phase geometry optimization
# memory str or list: Memory per core (e.g., "2500 mb")
# start str or list: Database name for simulation
# scratch_dir str or list: Where to store temporary files
# permanent_dir str or list: Where to store permanent files (e.g., results)
# charge str or list: NWChem charge directive
# geometry str or list: The .xyz file to load the initial geometry
# basis str or list: NWChem Basis set keyword
# maxIter str or list: Max # of iterations to optimize
# xyz str or list: Filename to save optimized geometry iterations to (e.g., xyz-01, xyz-02, etc.)
# xc str or list: NWChem xc Keyword for Exchange-Correlation Functional
# mult str or list: NWChem mult Keyword for Spin Multiplicity
# grid str or list: NWChem grid Keyword for Optimization Fidelity
# disp str or list: NWChem disp Keyword for dispersion correction
# taskGeomOptimize str or list: NWChem task directive for the geometry optimization (e.g., task dft optimize)
# titleFreq str or list: Title for the frequency calculation
# taskFreq str or list: NWChem task directive for the frequency calculation (e.g., task dft freq)
# titleSolve str or list: Title for the solvation energy calculation
# dielec str or list: COSMO dielectric constant
# minbem str or list: COSMO minbem optimization parameter
# ificos str or list: COSMO ificos optimization parameter
# do_gasphase str or list: COSMO do_gasphase parameter. If true, a gas phase geometry optimization is performed before implicit solvation.
# taskSolvEnergy str or list: NWChem task directive for solvated system energy calculation (e.g., task dft energy)
# inputScriptFolderTemplate str: Each unique input script gets its own folder, and this variable controls the name of the folders.
#                                (e.g., inputScriptFolderTemplate="TESTFOLDER" for 3 unique input scripts yields TESTFOLDER1, TESTFOLDER2, TESTFOLDER3)
class RedoxPotentialScript(NWChemInputConstructor):
    def __init__(
        self,
        titleGeomOptimizer: Union[str, list] = '"EMPTY"',
        memory: Union[str, list] = "512 mb",
        start: Union[str, list] = "EMPTY",
        scratch_dir: Union[str, list] = ".",
        permanent_dir: Union[str, list] = ".",
        charge: Union[str, list] = "EMPTY",
        geometry: Union[str, list] = "EMPTY",
        basis: Union[str, list] = "EMPTY",
        maxIter: Union[str, list] = "EMPTY",
        xyz: Union[str, list] = "EMPTY",
        xc: Union[str, list] = "EMPTY",
        mult: Union[str, list] = "EMPTY",
        grid: Union[str, list] = "EMPTY",
        disp: Union[str, list] = "EMPTY",
        taskGeomOptimize: Union[str, list] = "dft optimize",
        titleFreq: Union[str, list] = "EMPTY_FREQ",
        taskFreq: Union[str, list] = "dft freq",
        titleSolv: Union[str, list] = "EMPTY_SOLV",
        dielec: Union[str, list] = "EMPTY",
        minbem: Union[str, list] = "3",
        ificos: Union[str, list] = "1",
        do_gasphase: Union[str, list] = "False",
        taskSolvEnergy: Union[str, list] = "dft energy",
        inputScriptFolderTemplate: str = None,
    ):
        super().__init__()
        self.titleGeomOptimizer = titleGeomOptimizer
        self.memory = memory
        self.start = start
        self.scratch_dir = scratch_dir
        self.permanent_dir = permanent_dir
        self.charge = charge
        self.geometry = geometry
        self.basis = basis
        self.maxIter = maxIter
        self.xyz = xyz
        self.xc = xc
        self.mult = mult
        self.grid = grid
        self.disp = disp
        self.taskGeomOptimize = taskGeomOptimize
        self.titleFreq = titleFreq
        self.taskFreq = taskFreq
        self.titleSolv = titleSolv
        self.dielec = dielec
        self.minbem = minbem
        self.ificos = ificos
        self.do_gasphase = do_gasphase
        self.taskSolvEnergy = taskSolvEnergy

        self.inputScriptFolderTemplate = inputScriptFolderTemplate

        # Get all the relevant variables we want to write to the script
        # Store them in a convenient self.inputs variable
        for key, value in self.__dict__.items():
            self.inputs[key] = value
        # Because this class is built on a baseline class, it inherits the baseline class' variables
        # These are not directly related to the input script, so let's remove them.
        self.inputs.pop("inputs", None)
        self.inputs.pop("directiveList", None)
        self.inputs.pop("inputStr", None)
        self.inputs.pop("inputScriptFolderTemplate", None)
        self.inputs.pop("batch", None)

        # Determine if we are in batch mode (any provided value is a list)
        self.batch = any(isinstance(value, list) for value in self.inputs.values())

        if self.batch:
            # Generate all combinations and build scripts
            self.batch_generation()
        else:
            # Build a single script so directiveList exists (prevents "EMPTY")
            self.build_script_instructions(
                str(self.titleGeomOptimizer),
                str(self.memory),
                str(self.start),
                str(self.scratch_dir),
                str(self.permanent_dir),
                str(self.charge),
                str(self.geometry),
                str(self.basis),
                str(self.maxIter),
                str(self.xyz),
                str(self.xc),
                str(self.mult),
                str(self.grid),
                str(self.disp),
                str(self.taskGeomOptimize),
                str(self.titleFreq),
                str(self.taskFreq),
                str(self.titleSolv),
                str(self.dielec),
                str(self.minbem),
                str(self.ificos),
                str(self.do_gasphase),
                str(self.taskSolvEnergy),
            )
            # Populate the constructed script
            self.construct_input_script()

    def __str__(self):
        if not self.batch:
            return self.construct_input_script()
        else:
            self.batch_generation()
            return "\n\n".join(script for script in self.scripts)

    # If batch-mode is activated, this function generates all possible combinations of input scripts from the specified variables.
    # Generated input scripts are saved in self.scripts, which represents the entire batch.
    # Returns self.scripts list: All possible input scripts from the specified options stored in a list.
    def batch_generation(self):
        # Figures out all combinations of specified inputs
        # Later uses these combinations below to generate input scripts
        self.batch_combinations()

        self.scripts = []
        for inputScriptInd in range(len(self.inputScriptSpace)):
            inputScript = self.inputScriptSpace[inputScriptInd]
            inputScript.append(inputScriptInd + 1)
            self.build_script_instructions(*inputScript)
            scriptStr = self.construct_input_script()
            self.scripts.append(scriptStr)

    # This function assembles the ordered row-by-row input of any given input script.
    # Returns self.directiveList list: An ordered row-by-row input of a given input script.
    #                                  To construct the actual input script, self.directiveList is read from index 0 onwards
    #                                  where each index correponds to a directive in the input file (e.g., dft, title, geometry, etc.)
    def build_script_instructions(
        self,
        titleGeomOptimizer: str,
        memory: str,
        start: str,
        scratch_dir: str,
        permanent_dir: str,
        charge: str,
        geometry: str,
        basis: str,
        maxIter: str,
        xyz: str,
        xc: str,
        mult: str,
        grid: str,
        disp: str,
        taskGeomOptimize: str,
        titleFreq: str,
        taskFreq: str,
        titleSolv: str,
        dielec: str,
        minbem: str,
        ificos: str,
        do_gasphase: str,
        taskSolvEnergy: str,
        inputScriptTrialNumber: int = None,
    ):
        self.title_geom_optimize = dirs.TitleDirective()
        self.title_geom_optimize.set_option("title", f'"{titleGeomOptimizer}"')
        self.memory = dirs.MemoryDirective()
        self.memory.set_option("memory", memory)
        self.start = dirs.StartDirective()
        self.start.set_option("start", start)

        # If we are submitting a batch set of jobs, then we want each job trial to have its' own
        # permanent and scratch folders.
        if self.batch:
            if self.inputScriptFolderTemplate is not None:
                inputScriptFolder = self.inputScriptFolderTemplate + str(
                    inputScriptTrialNumber
                )
            else:
                inputScriptFolder = "inputScript" + str(inputScriptTrialNumber)

            self.scratch_dir = dirs.ScratchDirDirective()
            self.scratch_dir.set_option(
                "scratch_dir", os.path.join(scratch_dir, inputScriptFolder)
            )
            self.permanent_dir = dirs.PermanentDirDirective()
            self.permanent_dir.set_option(
                "permanent_dir", os.path.join(permanent_dir, inputScriptFolder)
            )
        else:
            if self.inputScriptFolderTemplate is not None:
                if inputScriptTrialNumber is not None:
                    inputScriptFolder = self.inputScriptFolderTemplate + str(
                        inputScriptTrialNumber
                    )
                else:
                    inputScriptFolder = self.inputScriptFolderTemplate
            else:
                if inputScriptTrialNumber is not None:
                    inputScriptFolder = str(inputScriptTrialNumber)
                else:
                    inputScriptFolder = "inputScript"
            self.scratch_dir = dirs.ScratchDirDirective()
            self.scratch_dir.set_option(
                "scratch_dir", os.path.join(scratch_dir, inputScriptFolder)
            )
            self.permanent_dir = dirs.PermanentDirDirective()
            self.permanent_dir.set_option(
                "permanent_dir", os.path.join(permanent_dir, inputScriptFolder)
            )

        self.charge = dirs.ChargeDirective()
        self.charge.set_option("charge", charge)
        self.geometry = dirs.GeometryDirective()
        self.geometry.set_option("load", geometry)
        self.basis = dirs.BasisDirective()
        self.basis.set_option("* library", basis)
        self.driver = dirs.DriverDirective()
        self.driver.set_option("MAXITER", maxIter)
        self.driver.set_option("XYZ", xyz)
        self.dft = dirs.DFTDirective()
        self.dft.set_option("xc", xc)
        self.dft.set_option("mult", mult)
        self.dft.set_option("grid", grid)
        self.dft.set_option("disp", disp)
        self.task_geom_optimize = dirs.TaskDirective()
        self.task_geom_optimize.set_option("task", taskGeomOptimize)
        self.title_freq = dirs.TitleDirective()
        self.title_freq.set_option("title", f'"{titleFreq}"')
        self.task_freq = dirs.TaskDirective()
        self.task_freq.set_option("task", taskFreq)
        self.title_solv = dirs.TitleDirective()
        self.title_solv.set_option("title", f'"{titleSolv}"')
        self.cosmo = dirs.COSMODirective()
        self.cosmo.set_option("dielec", dielec)
        self.cosmo.set_option("minbem", minbem)
        self.cosmo.set_option("ificos", ificos)
        self.cosmo.set_option("do_gasphase", do_gasphase)
        self.task_solv_energy = dirs.TaskDirective()
        self.task_solv_energy.set_option("task", taskSolvEnergy)

        self.directiveList = [
            self.title_geom_optimize,
            self.memory,
            self.start,
            self.scratch_dir,
            self.permanent_dir,
            self.charge,
            self.geometry,
            self.basis,
            self.driver,
            self.dft,
            self.task_geom_optimize,
            self.title_freq,
            self.task_freq,
            self.title_solv,
            self.cosmo,
            self.task_solv_energy,
        ]
