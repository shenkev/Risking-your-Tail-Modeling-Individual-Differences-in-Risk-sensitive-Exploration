# Risk-Aversive-Exploration

Accompanying code for the paper [Risking your Tail: Modeling Individual Differences in Risk-sensitive Exploration using Bayes Adaptive Markov Decision Processes](https://elifesciences.org/reviewed-preprints/100366#tab-content).

## Julia and Python

Julia: version 1.7.2 (packages in Project.toml)
Python: version 3.9 (packages in requirements.txt)

## Data

This project models data from the paper: [Striatal dopamine explains novelty-induced behavioral dynamics and individual variability in threat prediction](https://www.cell.com/neuron/fulltext/S0896-6273(22)00758-9). The original dataset can be found [here](https://datadryad.org/dataset/doi:10.5061/dryad.41ns1rnh2).

We start with bout-related data in the `raw_mice_data` folder:

- `frame_within.csv` (array of binary values whether each frame 135000 = 60 sec/minute * 150 minutes * 15 frame/sec has the mouse inside the 7cm radius or not)
- `frame_within_tail_behind.csv` (same but values are only 1 if it's tail-behind approach)
- `frame_within_tail_not_behind.csv` (same but values are only 1 if it's tail-infront approach)
- `bout_start.json` (dict from animal number to array of bout start frames - possible values from 1 to 135000)
- `bout_end.json` (dict from animal number to array of bout end frames - index-matched to `bout_start.json`)
- `bout_duration_raw.csv` (135000 dimension array of bout durations - most values are 0 since no bouts start during that frame, the indices in `bout_start.json`, subtract 1, are used to access bout durations in this file)

## Fitting Box Functions and Defining Phases

Run the notebook `./jupyter/phase_fitting.ipynb` to generate the cautious-to-confident and peak-to-steady-state phase boundaries in `./processed_mice_data/phase_params.json`.

Due to stochasticity in the box-fitting algorithm, the output may vary slightly from run to run. Use the existing data file `./processed_mice_data/phase_params.json` to reproduce the results in the paper.

## Generating ABCSMC Groundtruth and Visualizing Phase-wise Stats

Run the notebook `./jupyter/extract_groundtruth.ipynb` to generate the abcsmc groundtruth in the file `./processed_mice_data/gt_abc_stats.json` and visualize the ground truth data in terms of phases. This notebook will take the phase boundaries previously computed and compute the duration and frequency statistics in each phase.

## ABCSMC Fitting

Perform fitting with the following command, making sure to update the path to the groundtruth file. The `-p` runs the fitting on a 64-core (128 virtual core) machine.

```julia --project=. -p 128 ./scripts/fit_abc_animals_noisyor.jl [animal_number]```

To fit more than one animal serially (start the next animal once the previous one has completed), use the following script.

```./abcsmc_noisyor.sh animal_1 animal_2 animal_3 ...```

## Visualizing the Fit

To visualize the results of the fit, including,

- the posterior
- simulations of the best particles
- histograms of posterior marginal-means across animals

run the `./jupyter/visualize_abc_fits.ipynb` file. This notebook will also save `abcsmc_best_samples.json` which are the best particles of the fit for each animal. This files will be used in the recovery analysis.

## Recovery

Recovery analysis works as follows.

1. We perform the first ABCSMC fit on the animal data. This gives us a posterior distribution for each animal.
2. In the `./jupyter/visualize_abc_fits.ipynb` file we identify the best particle for each animal (ties are broken arbitrarily) and set these as targets for the recovery analysis. These particles are saved to `abcsmc_best_samples.json` which contains both the particles under the "theta" key and statistics of the corresponding simulations under the "x" key.
3. Note: the statistics are in "model space" rather than "animal space". E.g. duration takes values = 2.0, 3.0, 4.0.
4. We perform a second ABCSMC fit using `abcsmc_best_samples.json` as targets. Then run `./jupyter/visualize_abc_fits.ipynb` a second time to visualize the results of this recovery fit. This time, save `./data/SMC_fits.json` using the notebook which are all the particles in the final posterior population of the recovery fit. Save `abcsmc_best_samples.json` for a second time, which are the best matching particles in the recovery fit.
5. Run `./jupyter/recovery.ipynb` which opens `./data/SMC_fits.json` and `abcsmc_best_samples.json` and creates recovery plots to help assess the recovery.

`./jupyter/recovery.ipynb` contains previously computed recovery plot images.