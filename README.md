# Cellgroup - A library to cluster and analyse cells in wells

<div align="center">
  
[![Tests](https://github.com/vanderschaarlab/autoprognosis/actions/workflows/test_pr.yml/badge.svg)](https://github.com/vanderschaarlab/autoprognosis/actions/workflows/test_pr.yml)
[![Tests](https://github.com/vanderschaarlab/autoprognosis/actions/workflows/test_full.yml/badge.svg)](https://github.com/vanderschaarlab/autoprognosis/actions/workflows/test_full.yml)
[![Tests R](https://github.com/vanderschaarlab/autoprognosis/actions/workflows/test_R.yml/badge.svg)](https://github.com/vanderschaarlab/autoprognosis/actions/workflows/test_R.yml)
[![Tutorials](https://github.com/vanderschaarlab/autoprognosis/actions/workflows/test_tutorials.yml/badge.svg)](https://github.com/vanderschaarlab/autoprognosis/actions/workflows/test_tutorials.yml)
[![Documentation Status](https://readthedocs.org/projects/autoprognosis/badge/?version=latest)](https://autoprognosis.readthedocs.io/en/latest/?badge=latest)

[![](https://pepy.tech/badge/autoprognosis)](https://pypi.org/project/autoprognosis/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/vanderschaarlab/autoprognosis/blob/main/LICENSE)
[![about](https://img.shields.io/badge/about-The%20van%20der%20Schaar%20Lab-blue)](https://www.vanderschaar-lab.com/)
[![slack](https://img.shields.io/badge/chat-on%20slack-purple?logo=slack)](https://join.slack.com/t/vanderschaarlab/shared_invite/zt-1pzy8z7ti-zVsUPHAKTgCd1UoY8XtTEw)

</div>

![image](https://github.com/vanderschaarlab/autoprognosis/raw/main/docs/arch.png "AutoPrognosis")

## :key: Features

- :fire: Understand the best clustering method for your cells
- :balloon: Use Cython to cluster cells
- :cyclone: Link clusters over time
- :grey_question: Impute missing data in time 
- :sunny: Have a variety of already functions 

## :rocket: Installation

Then create the `Cellgroup` environment using:

```bash
conda env create -f Cellgroup.yml
```

Once the environment has been created, you can activate it and use `morphometrics` as described below.

```bash
conda activate Cellgroup
```

If you are on Mac OS or Linux install the following:

### Mac:

```bash
conda install -c conda-forge Cellgroup
```

### Linux:

```bash
conda install -c conda-forge Cellgroup
```

## Environment variables
The library can be configured from a set of environment variables.

| Variable       | Description                                                     |
|----------------|-----------------------------------------------------------------|
| `N_OPT_JOBS`     | Number of cores to use for hyperparameter search. Default : 1 |
| `N_LEARNER_JOBS` | Number of cores to use by inidividual learners. Default: all cpus      |
| `REDIS_HOST`     | IP address for the Redis database. Default 127.0.0.1            |
| `REDIS_PORT`     | Redis port. Default: 6379                                       |

_Example_: `export N_OPT_JOBS = 2` to use 2 cores for hyperparam search.


## Example applications
<table border="0">
<tr><td>


<img src="https://github.com/kevinyamauchi/morphometrics/raw/main/resources/surface_distance_measurement.gif"
width="300"/>

</td><td>

[Scikit_all_clusters](https://github.com/kevinyamauchi/morphometrics/blob/main/examples/surface_distance_measurement.ipynb)

</td></tr><tr><td>

<img src="https://github.com/kevinyamauchi/morphometrics/raw/main/resources/region_props_plugin.png"
width="300"/>

</td><td>

[Density_clusters](https://github.com/kevinyamauchi/morphometrics/blob/main/examples/measure_with_widget.py)

</td></tr><tr><td>

<img src="https://github.com/kevinyamauchi/morphometrics/raw/main/resources/object_classification.png"
width="300"/>

</td><td>

[Time_clustering](https://github.com/kevinyamauchi/morphometrics/blob/main/examples/object_classification.ipynb)

</td></tr><tr><td>

<img src="https://github.com/kevinyamauchi/morphometrics/raw/main/resources/mesh_object.png"
width="300"/>

</td><td>

[Clustering_analysis](https://github.com/kevinyamauchi/morphometrics/blob/main/examples/mesh_binary_mask.ipynb)

</td></tr><tr><td>


<img src="https://github.com/kevinyamauchi/morphometrics/raw/main/resources/mesh_object.png"
width="300"/>

</td><td>
  
[Time_clustering](https://github.com/kevinyamauchi/morphometrics/blob/main/examples/object_classification.ipynb)

</td></tr><tr><td>


<img src="https://github.com/kevinyamauchi/morphometrics/raw/main/resources/mesh_object.png"
width="300"/>

</td><td>

[Clustering_Predict](https://github.com/kevinyamauchi/morphometrics/blob/main/examples/mesh_binary_mask.ipynb)

</td></tr></table>
