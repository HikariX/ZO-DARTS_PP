for seed in 1 2 3
do
    for m in DARTS MileNAS ZO
    do
        for r in 1 2 3
        do
            # nohup python -u retrain_sampling_old.py --method ${m} --rand_seed ${seed} --dataset OrganAMNIST >./result_Others/${m}_organa_seed${seed}_round${r}.log 2>&1 &
            # nohup python -u retrain_sampling_old.py --method ${m} --rand_seed ${seed} --dataset OrganCMNIST >./result_Others/${m}_organc_seed${seed}_round${r}.log 2>&1 &
            # nohup python -u retrain_sampling_old.py --method ${m} --rand_seed ${seed} --dataset OrganSMNIST >./result_Others/${m}_organs_seed${seed}_round${r}.log 2>&1 &
            # nohup python -u retrain_sampling_old.py --method ${m} --rand_seed ${seed} --dataset PneumoniaMNIST >./result_Others/${m}_pneumonia_seed${seed}_round${r}.log 2>&1 &
            # nohup python -u retrain_sampling_old.py --method ${m} --rand_seed ${seed} --dataset OCTMNIST >./result_Others/${m}_oct_seed${seed}_round${r}.log 2>&1 &
            # nohup python -u retrain_sampling_old.py --method ${m} --rand_seed ${seed} --dataset BreastMNIST >./result_Others/${m}_breast_seed${seed}_round${r}.log 2>&1 &
            # nohup python -u retrain_sampling_old.py --method ${m} --rand_seed ${seed} --dataset TissueMNIST >./result_Others/${m}_tissue_seed${seed}_round${r}.log 2>&1 &
            # nohup python -u retrain_sampling_old.py --method ${m} --rand_seed ${seed} --dataset BloodMNIST >./result_Others/${m}_blood_seed${seed}_round${r}.log 2>&1 &
            # nohup python -u retrain_sampling_old.py --method ${m} --rand_seed ${seed} --dataset DermaMNIST >./result_Others/${m}_derma_seed${seed}_round${r}.log 2>&1 &
            nohup python -u retrain_sampling_old.py --method ${m} --rand_seed ${seed} --dataset PathMNIST >./result_Others/${m}_path_seed${seed}_round${r}.log 2>&1 &
        done
    done
done