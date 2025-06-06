
<h1 align="center">Data-Efficient Learning from Human Interventions for Mobile Robots</h1>

<h3 align="center"><b>ICRA 2025</b>
</h3>

<h3 align="center">
  <a href="https://metadriverse.github.io/pvp4real/"><b>Webpage</b></a> |
  <a href="https://github.com/metadriverse/pvp4real"><b>Code</b></a> |
  <a href="https://arxiv.org/pdf/2503.04969"><b>PDF</b></a>
</h3>

<p align="center">
  <img src="PVP4Real_Teaser.png" alt="PVP4Real" width="100%">
</p>

Compared to [PVP repo](https://metadriverse.github.io/pvp/), we include the simulated human experiments in this repo.




## Installation

TODO: `pvp4real` uses Python 3.7, but MetaUrban uses Python 3.9. Do I need to use 3.9 for MetaUrban integration?

```bash
# Clone the code to local machine
git clone https://github.com/pengzhenghao/pvp4real
cd pvp4real

# Create Conda environment
conda create -n pvp4real python=3.9
conda activate pvp4real

# Install dependencies
pip install setuptools==65.5.0 pip==21  # Fix gym installation issue
pip install wheel==0.38.0  # Fix gym installation issue
pip install -r requirements.txt
pip install -e .
pip install torch

# Install MetaDrive
# In case you need it, the MetaDrive commit we ran on is: c29cc37d30158fe70d963647b6c80dc814248f60
# Using latest MetaDrive should work:
pip install git+https://github.com/metadriverse/metadrive.git

# Install MetaUrbanw with ORCA (necessary for MetaUrban's expert policy)


```

Also, install MetaUrban in a separate directory *outside of this project*. You must install from source to use MetaUrban's expert policy (as it requires ORCA for trajectory generation).

```bash
git clone -b main --depth 1 https://github.com/metadriverse/metaurban.git
cd metaurban
pip install -e .
conda install pybind11 -c conda-forge
cd metaurban/orca_algo && rm -rf build
bash compile.sh && cd ../..
pip install stable_baselines3 imitation tensorboard wandb scikit-image pyyaml gdown
```

If you encounter an error when running MetaUrban that your environment using an outdated `libstdc++.so.6` which doesn't support the required `GLIBCXX_3.4.32` symbol version, you can try to update `libstdc++` via `conda` by running:

```bash
conda install -n pvp4real libstdcxx-ng=12.2.0
conda install --force-reinstall libstdcxx-ng -c conda-forge
strings $(g++ -print-file-name=libstdc++.so.6) | grep GLIBCXX_3.4.32
```

and ensure that the output is `GLIBCXX_3.4.32`.


## Launch Experiments

You can find the scripts to launch a batch of experiments in the [`./scripts`](./scripts) folder. Call them by:

```bash
bash scripts/metadrive_simhuman_pvp4real.sh
bash scripts/metadrive_simhuman_pvp.sh
bash scripts/metadrive_simhuman_haco.sh
bash scripts/metadrive_simhuman_ppo.sh
bash scripts/metadrive_simhuman_td3.sh
bash scripts/metadrive_simhuman_bc.sh
bash scripts/metadrive_simhuman_eil.sh
bash scripts/metadrive_simhuman_hgdagger.sh
```


Here is the example on how to launch PVP4Real:

```bash
python pvp/experiments/metadrive/train_pvp_metadrive_fakehuman.py \
--exp_name="pvp4real" \
--bc_loss_weight=1.0
```



---

Tips:

We evaluate frequently to get beautiful learning curves. You can change the evaluation frequency when debugging:

```python
model.learn(
    ...
    eval_freq=150,  # <<< Change this
    n_eval_episodes=50,  # <<< Change this
)
```

You can use `wandb` to monitor the training process:

```bash
nohup python pvp/experiments/metadrive/train_pvp_metadrive_fakehuman.py \
--exp_name=${EXP_NAME} \
--wandb \
--wandb_project=pvp2024 \
--wandb_team=drivingforce
```

> [!NOTE]  
> The BC, HGDagger and EIL experiments are not strictly following the paper.
> The BC data is not collected before training but instead collected and is expanding
> during the training process. 

## 📎 References

```latex
@article{peng2025data,
  title={Data-Efficient Learning from Human Interventions for Mobile Robots},
  author={Peng, Zhenghao and Liu, Zhizheng and Zhou, Bolei},
  journal={arXiv preprint arXiv:2503.04969},
  year={2025}
}
```
