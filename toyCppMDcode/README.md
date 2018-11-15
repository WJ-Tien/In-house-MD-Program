# One Dimension Molecular Dynamics Code

### Modify all the parameters in "PARMFILE"

### To compile:
	 make (executable binary file "run" will be generated)

### To run the MD code:
	 ./run input_params output_filename (2 argvs)

### To visualize the simulation:
	 ./visualizer.py output_file_from_MD output_mp4_name fps (3 argvs; fps = playback speed) 

### Examination of Boltzmann distribution
	 ./checkBoltzmannDist.py output_file_from_MD bin_width output_filename (3 argvs)

### To clean (e.g. modifications were made):
	 make clean (remove all .o files and executable binary file "run")
