# env
we directly use the environment of x-flux

``` bash
conda activate x-flux
```

have try it, but it seems that env not work

``` bash
conda create -n regionalFlux python==3.10
conda activate regionalFlux
# install diffusers locally
git clone https://github.com/huggingface/diffusers.git
cd diffusers

# reset diffusers version to 0.31.dev, where we developed Regional-Prompting-FLUX on, different version may experience different results
git reset --hard d13b0d63c0208f2c4c078c4261caf8bf587beb3b
pip install -e ".[torch]"
cd ..

# install other dependencies
pip install -U transformers sentencepiece protobuf PEFT

# clone this repo
git clone https://github.com/antonioo-c/Regional-Prompting-FLUX.git

# replace file in diffusers
cd Regional-Prompting-FLUX
cp transformer_flux.py ./diffusers/src/diffusers/models/transformers/transformer_flux.py

# additionally, if you want use PULID with Regional-Prompting-FLUX, follow the installation guide in [PULID](https://github.com/ToTheBeginning/PuLID) to install the necessary dependencies. Then,
# cd .. && git clone https://github.com/ToTheBeginning/PuLID.git
# cd Regional-Prompting-FLUX
# cp transformer_flux_pulid.py ../diffusers/src/diffusers/models/transformers/transformer_flux.py
```

# run

```bash
python quickStart.py 
```