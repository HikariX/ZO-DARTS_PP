for seed in 1 2 3
do
    # nohup python -u ZO-DARTS_SPARSE_anneal_med.py --data_path ../../nasbench201/dataset/organamnist.npz --rand_seed ${seed} --dataset OrganAMNIST --batch_size 256 --save_dir ./ZO+/ZO_SA_organa_+${seed}>organa.log 2>&1
    # nohup python -u ZO-DARTS_SPARSE_anneal_med.py --data_path ../../nasbench201/dataset/organcmnist.npz --rand_seed ${seed} --dataset OrganCMNIST --batch_size 256 --save_dir ./ZO+/ZO_SA_organc_+${seed}>organc.log 2>&1
    # nohup python -u ZO-DARTS_SPARSE_anneal_med.py --data_path ../../nasbench201/dataset/organsmnist.npz --rand_seed ${seed} --dataset OrganSMNIST --batch_size 256 --save_dir ./ZO+/ZO_SA_organs_+${seed}>organs.log 2>&1
    # nohup python -u ZO-DARTS_SPARSE_anneal_med.py --data_path ../../nasbench201/dataset/pneumoniamnist.npz --rand_seed ${seed} --dataset PneumoniaMNIST --batch_size 32 --save_dir ./ZO+/ZO_SA_pneumonia_+${seed}>pneumonia.log 2>&1
    # nohup python -u ZO-DARTS_SPARSE_anneal_med.py --data_path ../../nasbench201/dataset/octmnist.npz --rand_seed ${seed} --dataset OCTMNIST --batch_size 512 --save_dir ./ZO+/ZO_SA_oct_+${seed}>oct.log 2>&1
    # nohup python -u ZO-DARTS_SPARSE_anneal_med.py --data_path ../../nasbench201/dataset/breastmnist.npz --rand_seed ${seed} --dataset BreastMNIST --batch_size 16 --save_dir ./ZO+/ZO_SA_breast_+${seed}>breast.log 2>&1
    # nohup python -u ZO-DARTS_SPARSE_anneal_med.py --data_path ../../nasbench201/dataset/tissuemnist.npz --rand_seed ${seed} --dataset TissueMNIST --batch_size 512 --save_dir ./ZO+/ZO_SA_tissue_+${seed}>tissue.log 2>&1
    # nohup python -u ZO-DARTS_SPARSE_anneal_med.py --data_path ../../nasbench201/dataset/bloodmnist.npz --rand_seed ${seed} --dataset BloodMNIST --batch_size 128 --save_dir ./ZO+/ZO_SA_blood_+${seed}>blood.log 2>&1
    # nohup python -u ZO-DARTS_SPARSE_anneal_med.py --data_path ../../nasbench201/dataset/dermamnist.npz --rand_seed ${seed} --dataset DermaMNIST --batch_size 128 --save_dir ./ZO+/ZO_SA_derma_+${seed}>derma.log 2>&1
    nohup python -u ZO-DARTS_SPARSE_anneal_med.py --data_path ../../nasbench201/dataset/pathmnist.npz --rand_seed ${seed} --dataset PathMNIST --batch_size 512 --save_dir ./ZO+/ZO_SA_path_+${seed}>path.log 2>&1
done