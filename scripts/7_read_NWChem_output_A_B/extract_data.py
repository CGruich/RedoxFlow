from extract_redox_variables_example_function import RedoxPotentialMiner  

file_list = ['CO2.out']  # List of files to process
for file in file_list:
    miner = RedoxPotentialMiner(stdoutFilepath='CO2.out')
    miner.variableDF = {}
    miner.search()
