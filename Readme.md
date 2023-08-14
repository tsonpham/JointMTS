# Lightweight joint inversion of point-source moment-tensor and station-specific time shifts

**Thanh-Son Pham** (thanhson.pham@anu.edu.au)

You might need to install TensorFlow, obspy, netCDF to be able to run the notebook. 
Please see me if you want to run it on NCI Gadi (Australia).

Content of this packages:

- CPS.py: library file for function to generate Green's function with CPS program (Herrmann, 2023).
- UTIL.py: utinily file for plotting and other supporting function.
- MT.py: implementation of joint moment tensor inversion using TensorFlow functionalities.
- 19971122172035: Observed data example downloaded from North California Earthquake Data Center.
- GREENS: Pre-computed Green's functions for the event. If you need to generate new Green's functions, please install CPS program (downloaded [here](https://www.eas.slu.edu/eqc/eqccps.html) ) and replace the path to excecutable files in the header of the CPS.py file.
- SoCal.plain.txt: 1D velocity model for Southern California.
- JointMT-LVC.ipynb: Jupyter notebook demonstrating the utilization of the inversion code.