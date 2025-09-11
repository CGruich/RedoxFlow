import pandas as pd
import numpy as np
from os import path as osp
from typing import Union
from tqdm import tqdm

# ----------------------------
# HELPER FUNCTION
# Author: Cameron Gruich


# Check if a sub-string is inside a line string of a file when reading text data.
# Inputs: subStr --> Str, sub-string to search for
#         fileLineList --> List of Strings, all the text lines of a file (e.g., 3000-line text file makes 3000-size list)
#         startingLineInd --> Index position of fileLineList to start checking lines at.
#         reverseFile --> Whether to look at the text lines in reverse order or not.
# Returns: Tuple, (Matching Line to Sub-string, Line # of file the match occurs)
def line_substr_match(subStr, fileLineList, startingLineInd=0, reverseFile=False):
    # Ensure the index specified is properly specified as an integer
    assert type(startingLineInd) == int

    # Default return value for function
    lineDefault = None
    if reverseFile == True:
        for lineInd in range(0, startingLineInd)[::-1]:
            if subStr in fileLineList[lineInd]:
                return (fileLineList[lineInd], lineInd)
        return (lineDefault, 0)
    else:
        for lineInd in range(startingLineInd, len(fileLineList)):
            if subStr in fileLineList[lineInd]:
                return (fileLineList[lineInd], lineInd)
        return (lineDefault, 0)


# ----------------------------

# ----------------------------
# HELPER FUNCTION
# Author: Cameron Gruich


# Extract full text lines relevant to redox potential calculation, plus some extra information that may be useful later.
# Inputs: fileLineList --> List of Strings, all the text lines ofa file (e.g., 3000-line text file makes 3000-size list)
#         subStrList --> List of Strings, a list of identifying markers to select lines that have the relevant variables in them.
#         searchFromBottom --> List of bools, True means search the file in reverse order from the bottom, False means otherwise.
# Returns: Tuple, (List of Matching Lines, List of index integers corresponding to the line numbers of matching lines)
def extract_redox_potential_lines(
    fileLineList,
    subStrList=[
        "Optimization converged",
        "@",
        "vib:animation",
        "Temperature",
        "Zero-Point correction to Energy",
        "Thermal correction to Energy",
        "Thermal correction to Enthalpy",
        "Total Entropy",
        "- Translational",
        "- Rotational",
        "- Vibrational",
        "DFT Final Molecular Orbital Analysis",
        "COSMO solvation results",
        "sol phase energy",
        "COSMO energy",
        "Nuclear repulsion energy",
        "Exchange-Corr. energy",
        "Coulomb energy",
        "One electron energy",
    ],
    searchFromBottom=[
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
    ],
):
    # List of ints representing line numbers on where to start the search for each redox potential line.
    # Defaults to 0 at start of algorithm.
    startingIndList = [0 for ind in range(len(subStrList))]

    # Ensure that we did not leave specified values out to do the search. All lists should be same size
    assert len(subStrList) == len(searchFromBottom) == len(startingIndList)

    # Store matched lines here
    matchLineList = []

    # For each identifying marker,
    for subStrInd in range(len(subStrList)):
        # If the first marker is used,
        if subStrInd == 0:
            # Initialize the search.
            matchLine = line_substr_match(
                subStrList[subStrInd],
                fileLineList,
                startingLineInd=startingIndList[subStrInd],
                reverseFile=searchFromBottom[subStrInd],
            )
        else:
            # Continue the search from the line index of the last matched line in the file.
            matchLine = line_substr_match(
                subStrList[subStrInd],
                fileLineList,
                startingLineInd=startingIndList[subStrInd - 1],
                reverseFile=searchFromBottom[subStrInd],
            )
        matchLineList.append(matchLine[0])
        startingIndList[subStrInd] = matchLine[1]

    return (matchLineList, startingIndList)


# ----------------------------

# ----------------------------
# Extract variables from the full text lines relevant to redox potential calculation, plus some extra information that may be useful later.
# Author: Cameron Gruich


# Inputs: redoxPotentialLines --> List of Strings, the relevant text file lines containing redox potential variables
#         saveFilePath --> String, where to save the .csv of variables
#         recordTokenCases --> A list of tuples, specific text tokens (or datatypes) that represent variables we want to record, usually numbers or decimal points
#         endTokenCases --> A list of strings, specific text tokens that tell us when to stop recording variables per-line during the search
#         unitList --> A list of strings, Units corresponding to the relevant redox potential variables
#         nonNumericMatchLineIndices --> A list of integers, some of the extracted text lines are not relevant to redox potential and are extra information
#                                        So, remove these lines by index position.
# Returns: Tuple, (List of Matching Lines, List of index integers corresponding to the line numbers of matching lines)
def redox_potential_variables(
    redoxPotentialLines,
    saveFilePath,
    startSearchIndList=[9, 35, 35, 35, 35, 35, 35, 35, 35, 35, 27, 27, 27, 27, 27],
    recordTokenCases=[
        (int, "."),
        (int, "."),
        (int, "."),
        (int, "."),
        (int, "."),
        (int, "."),
        (int, "."),
        (int, "."),
        (int, "."),
        (int, "."),
        (int, "."),
        (int, "."),
        (int, "."),
        (int, "."),
        (int, "."),
    ],
    endTokenCases=["", "K", "", "", "", "", "", "", "", "", "", "", "", "", ""],
    unitList=[
        "System Energy (Hartree)",
        "Temperature (K)",
        "Zero-Point correction to Energy (kcal/mol)",
        "Thermal correction to Energy (kcal/mol)",
        "Thermal correction to Enthalpy (kcal/mol)",
        "Total Entropy (cal/mol-K)",
        "Translational Entropy (cal/mol-K) ",
        "Rotational Entropy (cal/mol-K)",
        "Vibrational Entropy (cal/mol-K)",
        "Solvated System Energy (Hartree)",
        "COSMO energy (Hartree)",
        "Nuclear repulsion energy (Hartree)",
        "Exchange-Corr. energy (Hartree)",
        "Coulomb energy (Hartree)",
        "One electron energy (Hartree)",
    ],
    nonNumericMatchLineIndices=[0, 2, 11, 12],
):
    # Ensure that we did not leave any specified values out. Lists should be equal size
    assert (
        len(startSearchIndList)
        == len(recordTokenCases)
        == len(endTokenCases)
        == len(unitList)
    )

    # Specify a token to skip over whitespace
    skipWhitespaceToken = " "
    # Specify a token for negative numbers
    negativeSignToken = "-"
    # Examine only the relevant text lines for redox potential variables
    redoxPotentialLines = [
        redoxPotentialLines[ind]
        for ind in range(len(redoxPotentialLines))
        if ind not in tuple(nonNumericMatchLineIndices)
    ]

    # Again, ensure lists of equal size
    assert len(redoxPotentialLines) == len(startSearchIndList)

    # Put recorded variables in this list
    recordedVariableList = []

    # For each line in the relevant redox potential lines,
    for lineInd in range(len(redoxPotentialLines)):
        # Strip the line of trailing characters (e.g., \n newline instances)
        strippedLine = redoxPotentialLines[lineInd].rstrip()
        # For any given relevant redox potential line, define the starting character to search the line by index
        startSearchInd = startSearchIndList[lineInd]
        # Define the tokens to record the variable number
        recordTokenCase = recordTokenCases[lineInd]
        # Define the token to stop recording the variable number
        endTokenCase = endTokenCases[lineInd]
        # Define the unit for the extracted variable
        unit = unitList[lineInd]
        # Start with an empty string variable, we will add characters as we read the variable from left-to-right.
        variable = ""

        # Keep a bool value to tell us when to start searching for the end token to terminate the search.
        terminateSearch = False
        # For each character on a line starting with the starting position to the end of the line,
        for characterInd in range(startSearchIndList[lineInd], len(strippedLine)):
            # Skip the character if it is whitespace and we are not actively trying to terminate the search
            if (
                strippedLine[characterInd] == skipWhitespaceToken
            ) and not terminateSearch:
                continue
            else:
                # Once we no longer scan over whitespace, we start recording numbers.
                # So, set terminateSearch to True to tell the algorithm to end the search after we can no longer record find numbers to record.
                terminateSearch = True
                # Our numbers give a sign (-), a decimal point (.), or an integer for each character.
                # If we are an integer,
                try:
                    intVal = int(strippedLine[characterInd])
                    if type(intVal) == recordTokenCase[0]:
                        # Add to variable to construct number
                        variable = variable + strippedLine[characterInd]
                # Else if we are a sign or decimal point,
                except:
                    # If our end token to find and terminate the search is whitespace, explicitly check for all possible whitespace characters or None-type value,
                    if endTokenCase == "" and (
                        strippedLine[characterInd].isspace()
                        or strippedLine[characterInd] == endTokenCase
                        or strippedLine[characterInd] == None
                        or strippedLine[characterInd] == " "
                    ):
                        # End search
                        break
                    # Otherwise if our end search token is not whitespace,
                    else:
                        if strippedLine[characterInd] == endTokenCase:
                            # End search
                            break
                    # If we identified a token to record, such as a number, sign, or decimal point,
                    if (recordTokenCase[1] == strippedLine[characterInd]) or (
                        negativeSignToken == strippedLine[characterInd]
                    ):
                        # Add to variable to construct number
                        variable = variable + strippedLine[characterInd]
        # Convert string number into a proper float
        variable = float(variable)
        # Add to list
        recordedVariableList.append(variable)

    # Save all the recorded variables to a dataframe,
    # The variables will have the proper units if lists are parallel.
    recordedVariableDF = pd.DataFrame(
        np.array(recordedVariableList).reshape(1, len(recordedVariableList)),
        columns=unitList,
    )
    recordedVariableDF.to_csv(saveFilePath)
    return recordedVariableDF


# ----------------------------
class NWChemTextMiner(object):
    def __init__(
        self,
        stdoutFilepath: Union[str, list] = None,
        stdoutFileExt: str = ".out",
        errorCheck: bool = True,
    ):
        self.stdoutFileExt = stdoutFileExt
        self.errorCheck = errorCheck

        if type(stdoutFilepath) is str:
            # Ensure file exists
            assert osp.isfile(stdoutFilepath)
        elif type(stdoutFilepath) is list:
            for fileName in stdoutFilepath:
                assert osp.isfile(fileName)

        self.stdoutFilepath = stdoutFilepath
        # Store the extracted variables as a dataframe. Default to None
        self.variableDF = None
        # Stores filenames of files that encountered errors or did not complete
        self.errFiles = []

    def check_error(
        self,
        stdoutFile: str,
        successLine: str = "                                      AUTHORS",
        lineSearchLimit: int = 200,
    ):
        # Try to open the file,
        try:
            with open(stdoutFile, "r") as fileObj:
                data = fileObj.readlines()
            # Look through the file from the bottom, in reverse order,
            counter = 0
            for line in reversed(data):
                # If simulation was successful,
                if successLine in line:
                    # No error in simulation found
                    return False
                else:
                    counter = counter + 1
                # If we have searched enough text lines and do not find simulation termination,
                if counter == lineSearchLimit:
                    # Stop searching
                    break
            # Error/incomplete simulation found
            self.errFiles.append(stdoutFile)
            return True
        # If we cannot open the file,
        except:
            self.errFiles.append(stdoutFile)
            return True

    def catch_errors(self):
        # If a single file,
        if type(self.stdoutFilepath) is str:
            self.check_error(self.stdoutFilepath)
        # If a list of files,
        elif type(self.stdoutFilepath) is list:
            for filepath in tqdm(self.stdoutFilepath):
                isError = self.check_error(filepath)
                if isError:
                    self.stdoutFilepath = [
                        filepathName
                        for filepathName in self.stdoutFilepath
                        if filepathName != filepath
                    ]


class RedoxPotentialMiner(NWChemTextMiner):
    def __init__(
        self,
        stdoutFilepath: Union[str, list],
        stdoutFileExt: str = ".out",
        errorCheck: bool = True,
    ):
        super().__init__()
        # By convention, the full filepath is recommended
        # (e.g., /full/filepath/to/file.txt, not ./file.txt)
        self.stdoutFilepath = stdoutFilepath
        self.stdoutFileExt = stdoutFileExt
        self.errorCheck = errorCheck

        if self.errorCheck:
            # Removes failed simulations from self.stdoutFilepath and saves it to a list in self.errFiles
            self.catch_errors()

    def __str__(self):
        assert self.variableDF is not None
        # If returning a dict or dataframe, this can be converted to a string easily
        return str(self.variableDF)

    def return_errors(self):
        return self.errFiles

    def search(self):
        if type(self.stdoutFilepath) is str:
            with open(self.stdoutFilepath) as fileObj:
                matchLinesAndIndices = extract_redox_potential_lines(
                    fileObj.readlines()
                )
                matchLines = matchLinesAndIndices[0]
                head, tail = osp.split(self.stdoutFilepath)
                self.variableDF[tail] = redox_potential_variables(
                    redoxPotentialLines=matchLines,
                    saveFilePath=osp.join(
                        head, tail.rstrip(self.stdoutFileExt) + "_redox.csv"
                    ),
                )
                return self.variableDF
        if type(self.stdoutFilepath) is list:
            self.variableDF = {}
            for filepath in tqdm(self.stdoutFilepath):
                with open(filepath) as fileObj:
                    matchLinesAndIndices = extract_redox_potential_lines(
                        fileObj.readlines()
                    )
                    matchLines = matchLinesAndIndices[0]
                    head, tail = osp.split(filepath)
                    variables = redox_potential_variables(
                        redoxPotentialLines=matchLines,
                        saveFilePath=osp.join(
                            head, tail.rstrip(self.stdoutFileExt) + "_redox.csv"
                        ),
                    )
                    self.variableDF[tail] = variables
            return self.variableDF

