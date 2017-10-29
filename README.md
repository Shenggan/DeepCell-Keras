# Deep Cell

Reimplement [Deep Cell](https://github.com/CovertLab/DeepCell) with tensorflow.

## Pre

Download Dataset:

```shell
./get_dataset.sh
```

Prepare the python enviroment:

```shell
pip install -r requirements.txt
```

or

```shell
conda env create -f environment.yml
```

### Usage

```shell
python launch.py pipeline
```

### Tips

We use "channels_first" date format in keras, you should config in `~/.keras/keras.json`.

### Thanks

[DeepCell](https://github.com/CovertLab/DeepCell) and [deepcell-tf](https://github.com/vanvalen/deepcell-tf)
