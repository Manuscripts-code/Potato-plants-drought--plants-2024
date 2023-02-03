## Readme linked to version v0.1

### Commands
Train CNN:
```
python main.py -c configs/conv/config_autoencoder_hyp.json -m train_conv
```

Train CNN with changed set validation split to zero:
```
python main.py -c configs/conv/config_autoencoder_hyp.json -m train_conv -vs 0
```

Fine-tune already trained CNN with validation split set to zero:
```
python main.py -r 31f63e986396487cbb0b28526bbb6d1d -m train_conv -vs 0.0 --learning_rate 0.00001
```

Test CNN:
```
python main.py -r edcaa45741d14b939310fa561b4b8eb3 -m test_conv
```

Train and test CNN:
```
python main.py -c configs/conv/config_autoencoder_hyp.json -m train_test_conv
```

Train traditional model (SVM):
```
python main.py -c configs/trad/config_svm.json -m train_trad
```

Test traditional model (SVM):
```
python main.py -r 0621cdf2506d4303ae7ecfbe07747c8e -m test_trad
```

Settings from above coppied from vscode launch file:
```
// "args": ["-c","configs/conv/config_autoencoder_hyp.json", "-m", "train_conv"],
// "args": ["-c","configs/conv/config_autoencoder_hyp.json", "-m", "train_conv", "-vs", "0"],
// "args": ["-r", "31f63e986396487cbb0b28526bbb6d1d", "-m", "train_conv", "-vs", "0.0", "--learning_rate", "0.00001"],
// "args": ["-r", "0621cdf2506d4303ae7ecfbe07747c8e", "-m", "test_conv"]
// "args": ["-c","configs/conv/config_autoencoder_hyp.json", "-m", "train_test_conv"],
// "args": ["-c","configs/trad/config_svm.json", "-m", "train_trad"]
// "args": ["-r", "0621cdf2506d4303ae7ecfbe07747c8e", "-m", "test_trad"]
```


### Run mlflow server to observe experiments
```
mlflow server -h 0.0.0.0 -p 8000 --backend-store-uri experiments/
```
The experiments excesable at <http://localhost:8000/>

