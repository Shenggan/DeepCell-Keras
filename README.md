# Deep Cell

Reimplement [Deep Cell](https://github.com/CovertLab/DeepCell) with Keras and [Horovod](https://github.com/uber/horovod). The paper of Deep Cell is [here](https://github.com/Shenggan/DeepCell-Keras/raw/master/PAPER/deepcell.pdf).

## Dataset

Download the [Dataset](http://138.68.43.52/DATA.tar.gz):

```shell
./get_dataset.sh
```

## Enviroment

If you use pip to manage your Python enviroment:

```shell
pip install -r requirements.txt
```

If you use conda as your Python enviroment:

```shell
conda env create -f environment.yml
```

### Usage

The sub-command can be 

1. data_prepare
2. train
3. validation
4. test

Or you can use pipeline as sub-command to run all 4 steps.

```shell
python launch.py pipeline
```

### Tips

* We use **"channels_first"** date format in Keras, you should config in `~/.keras/keras.json`.

* If you want to run a distributed training, you must install [Horovod](https://github.com/uber/horovod) and run

	```shell
	mpirun -np 4 python deepcell/train.py -e 20 --dist 1
	```

### Thanks

[DeepCell](https://github.com/CovertLab/DeepCell) and [deepcell-tf](https://github.com/vanvalen/deepcell-tf)
