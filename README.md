# One Dimension Molecular Dynamics Code

## Modify all the parameters in "PARMFILE"

### To compile:
	#### make (executable binary file "run" will be generated)

### To run the MD code:
	#### ./run input_params output_filename (2 argvs)

### To visualize the simulation:
	#### ./visualizer.py data_to_be_visualized output_mp4_name fps (3 argvs, argv[1] = output file from ./run; fps = playback speed) 

### To clean (e.g. modifications were made):
	#### make clean(remove all .o files and executable binary file "run")
