@REM ___ Krka stratify configs (correct plant depentant split) ___
@REM python main.py -c configs/krka/stratify/krka_stratify_5.json -m train_test
@REM python main.py -c configs/krka/stratify/krka_stratify_4.json -m train_test
@REM python main.py -c configs/krka/stratify/krka_stratify_3.json -m train_test
@REM python main.py -c configs/krka/stratify/krka_stratify_2.json -m train_test
@REM python main.py -c configs/krka/stratify/krka_stratify_1.json -m train_test
@REM python main.py -c configs/krka/stratify/krka_stratify_54.json -m train_test
@REM python main.py -c configs/krka/stratify/krka_stratify_543.json -m train_test
@REM python main.py -c configs/krka/stratify/krka_stratify_5432.json -m train_test
@REM python main.py -c configs/krka/stratify/krka_stratify_54321.json -m train_test

@REM ___ Krka stratify configs + added l1 penalty(correct plant depentant split) ___
@REM python main.py -c configs/krka/stratify_penalty/krka_stratify_5.json -m train_test
@REM python main.py -c configs/krka/stratify_penalty/krka_stratify_4.json -m train_test
@REM python main.py -c configs/krka/stratify_penalty/krka_stratify_3.json -m train_test
@REM python main.py -c configs/krka/stratify_penalty/krka_stratify_2.json -m train_test
@REM python main.py -c configs/krka/stratify_penalty/krka_stratify_1.json -m train_test
@REM python main.py -c configs/krka/stratify_penalty/krka_stratify_5432.json -m train_test

@REM ___ Krka random (incorrect plant independat split) ___
@REM python main.py -c configs/krka/random/krka_random_5.json -m train_test
@REM python main.py -c configs/krka/random/krka_random_4.json -m train_test
@REM python main.py -c configs/krka/random/krka_random_3.json -m train_test
@REM python main.py -c configs/krka/random/krka_random_2.json -m train_test
@REM python main.py -c configs/krka/random/krka_random_1.json -m train_test
@REM python main.py -c configs/krka/random/krka_random_54.json -m train_test
@REM python main.py -c configs/krka/random/krka_random_543.json -m train_test
@REM python main.py -c configs/krka/random/krka_random_5432.json -m train_test
@REM python main.py -c configs/krka/random/krka_random_54321.json -m train_test


@REM ___ Krka dumb (random mix of target classes) ___
@REM python main.py -c configs/krka/dumb/krka_dumb_5.json -m train_test


@REM ___ Krka biased (plant dependant split, but biased underlying distribution) ___
@REM python main.py -c configs/krka/bias/krka_bias_54321_im.json -m train_test
@REM python main.py -c configs/krka/bias/krka_bias_54321_treat.json -m train_test
