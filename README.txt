This reposity forks the original repository written by Mira Liu (working with
Stephen Meyer, Ritoban Basu Thakur, and Zhaodi Pan) to simulate an FTS. 

The fork adds new functionality to simulate both the FTS and the coupling
optics used for the ACT passband measurements.

Major code additions include a ray-trace (helped with code from Grace Chesmore)
to help simulate the coupling optics which were bolted onto the FTS and used in
the ACT passband measurements. Additional modifications include steps to store
the path length of the rays and compute the interferograms for all frequencies
at once, as well as changes in how the detector plane is segmented and how the
rays are weighted assuming propagation into a detector feedhorn. 

The main results from the simulation tailored to the ACT passband measurements
compromise

(1) The Frequency shift found for this FTS dependent on the detector position
relative to the FTS focus.

(2) The transfer function in amplitude over the measured frequency range,
dependent on the detector position in the focal plane as well the amount of
defocus (shift along boresight) of optical system.

These effects have been quantified and informed the final ACT passband
measurements and systematic uncertainties.

An example showing how to trace rays through this optical system and check the
optical systematics found is shown in the example notebook
Example_Simulation_20260128.ipynb.

