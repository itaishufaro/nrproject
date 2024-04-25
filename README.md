# Comparing Q-Learning variants with Upper Confidence Bounds

The algorithm and the baselines are implemented in the folder `algorithms/`. The folder `config/` contains the parameters defining the experiments.

* Requirements:
    * Python 3.7
    * [`rlberry`](https://github.com/rlberry-py/rlberry) version 0.1
    * pyyaml

* Create and activate conda virtual environment (optional)

```bash
$ conda create -n ucbmq_env python=3.7
$ conda activate ucbmq_env
```

* Install requirements

```bash
$ pip install 'rlberry[full]==0.1'
$ pip install pyyaml
```

* Run and plot

```bash
$ python run.py config/experiment.yaml --n_fit=8
$ python plot.py
```


Original repository: https://github.com/omardrwch/ucbmq_code