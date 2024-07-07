This folder contains files to run the FS-SNS:

Before running unzip the Filter_Methods and Simulation_Results_20240203 Folder

The files to run the SNS with different feature selection methods is as follows:
	1. Base_SNS_Control_File to run the Base SNS
	2. FS_SNS_Control_File.py to run the FS-SNS with FSV1

The other files in the folder include:
	- Graph_Creation.py: Graph create files for large and small networks
	- FeatureExtraction.py: A feature extraction file to extract the features from the graph.ml files
	- Evaluation.py: An evaluation file
	- Large_Graph_Extraction_V2.py: file to extract Large Graphs

For a deeper understanding of the code, comments are provided to understand how the interpret the code.
The folders that are within include:
	- Target graphs in the "Test_Graphs" folder
	- Results in the "Simulation_Results" folder and feature ranking results in the "FSV1_Results" and the "FSV2_Results" folders
	- The extracted features from each network is in the "FeatureExtraction" folder
	- Finally the filtering methods used in the feature ranking code are in the "Filter_Methods" folder 