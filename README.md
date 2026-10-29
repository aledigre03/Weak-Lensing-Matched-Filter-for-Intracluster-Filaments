# Weak-Lensing-Matched-Filter-for-Intracluster-Filaments

This Python script implements a weak-lensing matched filter to identify intracluster filaments in simuations' snapshots. It combines IllustrisTNG snapshots, κTNG lensing maps, and galaxy catalogue information from the Hyper Suprime-Cam Subaru Strategic Program (HSC) to test filament detection under realistic observational conditions.

The program selects a halo from an IllustrisTNG100 snapshot and project dark matter particles onto a two-dimensional surface-density grid. This distribution is converted into convergence maps for different source redshifts. Via Fast Fourier transformation it derives the two shear components. The result is the lensing signal of the halo and its surrounding structure.

The workflow expects an HSC catalogue as an input to supply galaxy positions, source redshifts, and shape-measurement uncertainties. Quality cuts are introduced to reproduce a clean field of view, such procedure includes the selection of non-overlapping sky windows with adequate coverage. The creation of 1000 uncorrelated line of sights requires resampling the patterns through rotations and reflections. Background galaxies contribute to the measurement.

Noise modelling includes unrelated large-scale structure, intrinsic galaxy ellipticities, and measurement errors. Dark-matter-only κTNG convergence and shear maps supply the line-of-sight lensing contribution at the assigned redshift source planes. Each mock source inherits its HSC lensing weight and measurement uncertainty, while an independent intrinsic ellipticity is randomly drawn. 

The filter requires an analytical power spectrum, calculated using CCL and CAMB with the weighted HSC source-redshift distribution. The template is mirrored to reduce boundary effects, Fourier transformed, divided by the total noise power, and transformed back into real space.

The application of the filter follows a geometrical procedure where the filter is rotated around the halo to produce tangential and cross-component signal curves. An ensemble of 1,000 noise realizations estimates uncertainties and angular covariance, allowing signal-to-noise ratios to be calculated. A separate single observation combines the snapshot signal with large-scale structure and shape noise. A circular cut in the inner region of the field is applied to avoid contamination from the halo's shear signal.

Finally, emcee samples the filament amplitude and width parameters along the strongest positive detected angular direction. The script saves numerical results, diagnostic plots, posterior distributions, and checkpoints.

AI Disclaimer:

I have made use of AI for writing this code, particularly when implementing the necessary cuts and rotations for the Subaru HSC data used in the analysis. The code was originally contained in a Python notebook, and I used large language model to help convert it into a Python script.

I also used AI as a learning tool to understand how to parallelize the code across multiple CPU cores. This was my first experience with this computational challenge, and ChatGPT was particularly useful for learning how to implement it.

I have reviewed the code in its entirety and confirm that it implements the workflow described previously. However, there may still be mistakes that I have overlooked. Please contact me with any questions, concerns, or suggestions, and I will reply promptly.
