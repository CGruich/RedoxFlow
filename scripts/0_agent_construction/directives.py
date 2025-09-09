# ----------------------------
# An NWChem input script consists of a series of 'directives', which could be either single-line or indented multi-line blocks of text
# For example, there is a 'geometry directive', a 'basis directive', etc.


# Here, we define a generic directive template that we can use to modularly build custom NWChem input scripts.
# directiveName str: The title of the directive. For example, a geometry directive would have the title 'geometry' because that's the keyword in NWChem.
# directiveOptions list: All the options to code in a multi-line directive. For example, one could pass directiveOptions=["xc", "grid"] to a DFT directive.
# innerIndenation str: Default indentation pattern for commands inside a multi-line directive block.
#
# Returns self.directiveStr str: This is a multi-line string representing the entire NWChem directive generated.
class NWChemDirective(object):
    def __init__(
        self,
        directiveName: str = None,
        directiveOptions: list = None,
        innerIndentation: str = " ",
    ):
        self.directiveName = directiveName
        self.directiveOptions = directiveOptions
        self.innerIndentation = innerIndentation

        self.directiveCommands = {}
        if self.directiveOptions is not None:
            if len(self.directiveOptions) > 0:
                for directiveOption in self.directiveOptions:
                    assert type(directiveOption) is str
                self.command_dictionary()

        if self.directiveName is None:
            self.directiveStr = "EMPTY"
        else:
            self.directiveStr = f"{self.directiveName}"

    def __str__(self):
        self.build_input_block()
        return self.directiveStr

    def command_dictionary(self):
        if len(self.directiveOptions) == 1:
            self.directiveCommands = {self.directiveOptions[0]: None}
        else:
            self.directiveCommands = dict.fromkeys(self.directiveOptions)

    def set_option(self, directiveOption: str, directiveValue: str):
        if directiveOption in self.directiveCommands:
            self.directiveCommands[directiveOption] = directiveValue
            self.build_input_block()
        elif directiveOption == self.directiveName:
            self.directiveStr = f"{self.directiveStr} {directiveValue}"
        else:
            Exception(
                f"\nERROR WITH SPECIFYING {self.directiveName} DIRECTIVE: {directiveOption} is not a valid option.\n"
            )

    def build_input_block(self):
        if self.directiveOptions is not None:
            self.directiveStr = f"{self.directiveName}"
            for option in self.directiveOptions:
                directiveValue = self.directiveCommands[option]
                optionStr = f"\n{self.innerIndentation}{option} {directiveValue}"
                self.directiveStr = self.directiveStr + optionStr
            self.directiveStr = self.directiveStr + "\nend"
            return self.directiveStr
        else:
            return self.directiveStr


# ----------------------------


# ----------------------------
# Here, we define some common input script directives.
# THESE DIRECTIVES ARE INTENDED TO BE CALCULATION-AGNOSTIC. They are 'primitives' that we can include in custom calculation classes (e.g., redox potential).
# In other words, the "GeometryDirective" class itself will not be tailored to any specific calculation.
# Directive options can be added as-needed to extend the functionality of these directives.
# Geometry Directive
class GeometryDirective(NWChemDirective):
    def __init__(
        self, directiveName: str = "geometry", directiveOptions: list = ["load"]
    ):
        super().__init__(directiveName=directiveName, directiveOptions=directiveOptions)


# Basis Directive
class BasisDirective(NWChemDirective):
    def __init__(
        self, directiveName: str = "basis", directiveOptions: list = ["* library"]
    ):
        super().__init__(directiveName=directiveName, directiveOptions=directiveOptions)


# Driver Directive
class DriverDirective(NWChemDirective):
    def __init__(
        self, directiveName: str = "driver", directiveOptions: list = ["MAXITER", "XYZ"]
    ):
        super().__init__(directiveName=directiveName, directiveOptions=directiveOptions)


# DFT Directive
class DFTDirective(NWChemDirective):
    def __init__(
        self,
        directiveName: str = "dft",
        directiveOptions: list = ["xc", "mult", "grid", "disp"],
    ):
        super().__init__(directiveName=directiveName, directiveOptions=directiveOptions)


# COSMO Directive
class COSMODirective(NWChemDirective):
    def __init__(
        self,
        directiveName: str = "cosmo",
        directiveOptions: list = ["dielec", "minbem", "ificos", "do_gasphase"],
    ):
        super().__init__(directiveName=directiveName, directiveOptions=directiveOptions)


# Title Directive
class TitleDirective(NWChemDirective):
    def __init__(self, directiveName: str = "title"):
        super().__init__(directiveName=directiveName, directiveOptions=None)


# Memory Directive
class MemoryDirective(NWChemDirective):
    def __init__(self, directiveName: str = "memory"):
        super().__init__(directiveName=directiveName, directiveOptions=None)


# Start Directive
class StartDirective(NWChemDirective):
    def __init__(self, directiveName: str = "start"):
        super().__init__(directiveName=directiveName, directiveOptions=None)


# Scratch Directory Directive
class ScratchDirDirective(NWChemDirective):
    def __init__(self, directiveName: str = "scratch_dir"):
        super().__init__(directiveName=directiveName, directiveOptions=None)


# Permanent Directory Directive
class PermanentDirDirective(NWChemDirective):
    def __init__(self, directiveName: str = "permanent_dir"):
        super().__init__(directiveName=directiveName, directiveOptions=None)


# Charge Directive
class ChargeDirective(NWChemDirective):
    def __init__(self, directiveName: str = "charge"):
        super().__init__(directiveName=directiveName, directiveOptions=None)


# Task Directive
class TaskDirective(NWChemDirective):
    def __init__(self, directiveName: str = "task"):
        super().__init__(directiveName=directiveName, directiveOptions=None)


