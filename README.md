# GWtuna
Trawling through the data to find gravitational waves (or fish)!

GWtuna is a fast gravitational-wave search prototype built on Optuna (optimisation software library) and JAX (accelerator-orientated array computation library). Using Optuna, we introduce black box optimisation algorithms and evolutionary strategy algorithms to the gravitational-wave community. Tree-structured Parzen Estimator (TPE) and Covariance Matrix Adaption Evolution Strategy (CMA-ES) have been used to create the first template bank free search and used to identify binary neutron star mergers. 

For a GWtuna example, see 'GWtunaBNSSearchPrototype' python script. GWtuna requires JAX implementation on a GPU and installation of revelant modules stated in this Python script. Note: the Ripple module was forked and the '@jit' functions removed. 

See the 'paper' folder, for the results showcased in the paper. This folder contains: 
1) 'outputfiles' - GWtuna outputs which are: successful injections, failed injections, and a log file. 
2) 'plots' - The plots showcased in the paper and the Juypter notebook used to create them. 
3) 3 GWtuna Python scripts used to gather results and the Advanced LIGO O4 PSD. 
