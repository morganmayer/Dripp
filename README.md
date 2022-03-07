# Dripp
DRop-In Property Prediction

Property prediction tool for Drop-in fuels using FTIR spectra as input predictors.

## Scripts and purpose

### compile_data.ipynb: 

Takes neat property csv file and blend volume composition file to export compiled dataframe csv files with the spectra, properties, and identification information.

### blend_property_correlations.py
Contains a python class for calculating properties for blends based on volume fraction. 

### make_artificial_spectra.ipynb

Creates and export artificial FTIR spectra for binary blends based on polyfits of absorbances at each wavenumber. Uses real spectra and volume fractions to interpolate absorbances for a requested additive volume fraction.

### data_viz.ipynb
Creates data visualizations of property and spectra patterns with whole dataset.

## Folders and organization

### data

#### spectra
raw spectra csv files for real experimental FTIR spectra for neat and blend fuels.

#### interpolated_spectra
artificial spectra created from make_artificial_spectra.ipynb

#### interpolated_spectra_analysis

contains csv files of R2 correlation fits of polyfits at each wavenumber and percent differences in absorbance agreement for artificial spectra made at compositions there is real spectra for. This is to analyze the performance of the polyfit method to synthesize FTIR spectra. Plots are stored in data/interpolated_spectra_analysis.

### plots

#### spectra_plots
contains plots of all spectra real and artificial.

#### interpolated_spectra_analysis

contains csv files of R2 correlation fits of polyfits at each wavenumber and percent differences in absorbance agreement for artificial spectra made at compositions there is real spectra for. This is to analyze the performance of the polyfit method to synthesize FTIR spectra. Plots of this data is stored in data/interpolated_spectra_analysis.

### input_files

Contains input files in yaml format for model comparison runs.

### hyperparameter_searches

Contains hyperparameter search space arrays in yaml format for model comparison runs.

## Workflow

### Make data files from raw data

Run compile_data.ipynb to create csv files with properties, spectra, and ID columns organized for neat fuels and fuel blends. It will use neat_properties.csv and blend_composition.csv and include all fuels from both files. It uses blend_property_correlations.py to calculate properties for fuel blends.


### Make artificial spectra via polynomial fit interpolation at each wavenumber

Run make_artificial_spectra.ipynb to create artificial spectra and export them to data/interpolated_spectra. Composition data is appended to blend_composition.csv file.

Also can make artificial spectra for blends there already exists real experimental spectra for. Agreement of real and artificial spectra is analyzed via $R^2$ fits, residuals, and percent differences at each wavenumber. These performance metrics are exported as csv files to data/interpolated_spectra_analysis and plotted and saved to plots/interpolated_spectra_analysis. Also stored here are plots for absorbance vs. additive volume fraction for requested wavenumbers.

### Recompile dataframe files to include artificial spectra

If artificial spectra needs to be added to compiled dataframes, re-run compile_data.ipynb.

