# To do
- Load manifold without loading controls?
- Correlation between latent trajectories and true latent trajectories
- Variable trial duration with mask? (automatically ignores nans?)
- SVD to determine the initial decoder weights? (log map?)
- Automatically determine rank (start high)
  - Could circumvent the time PC alignment to X_i
  - Check Francesco's paper

# Done
- Fit manifold and controls separately
- Try variable number of conditions bias?
- PoissonLL fitting
- Try alternate optimization of manifold and controls?
- The controls_time_dim parameter should be a proportion (float) of time_dim?
