# Foreground subtraction and power spectrum computing pipeline for SKA SDC3a

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10124117.svg)](https://doi.org/10.5281/zenodo.10124117)

To run the pipeline, first create a 'config.ini' file in current dir:

```
[DEFAULT]
img_file = /dir/to/SKA/SDC3/ZW3.msw_image.fits
test_img_file = /dir/to/SKA/SDC3/TestDataset.msw_image.fits
beam_file = /dir/to/SKA/SDC3/station_beam.fits
true_pk2d_file = /dir/to/SKA/SDC3/TestDatasetTRUTH_166MHz-181MHz.data
test_pk2d_file = ./TianlaiTest_166MHz-181MHz.data
test_err_file = ./TianlaiTest_166MHz-181MHz_errors.data
```

Run the pipeline as follows:

```
> python pipe.py
```

Run pipeline for the Test Dataset as follows:

```
> python test_pipe.py
```

# TODO:

1. Try various foreground subtraction method;
2. Check unit conversion;
3. Power spectrum error estimation;
4. PSF deconvolution.