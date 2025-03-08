for s in 1 2 3
do
    for seed in 1 2 3
    do
        # nohup python -u ZO-DARTS_SAMVR.py --rand_seed ${seed} --data_path ../../nasbench201/dataset/organamnist.npz --dataset OrganAMNIST --size ${s} --batch_size 256 --save_dir ./Penalty15_percentile/ZO_SAP_organa_+${seed}_constraint${s}>organa.log 2>&1
        # nohup python -u ZO-DARTS_SAMVR.py --rand_seed ${seed} --data_path ../../nasbench201/dataset/organcmnist.npz --dataset OrganCMNIST --size ${s} --batch_size 256 --save_dir ./Penalty15_percentile/ZO_SAP_organc_+${seed}_constraint${s}>organc.log 2>&1
        # nohup python -u ZO-DARTS_SAMVR.py --rand_seed ${seed} --data_path ../../nasbench201/dataset/organsmnist.npz --dataset OrganSMNIST --size ${s} --batch_size 256 --save_dir ./Penalty15_percentile/ZO_SAP_organs_+${seed}_constraint${s}>organs.log 2>&1
        # nohup python -u ZO-DARTS_SAMVR.py --rand_seed ${seed} --data_path ../../nasbench201/dataset/octmnist.npz --dataset OCTMNIST --size ${s} --batch_size 512 --save_dir ./Penalty15_percentile/ZO_SAP_oct_+${seed}_constraint${s}>oct.log 2>&1
        # nohup python -u ZO-DARTS_SAMVR.py --rand_seed ${seed} --data_path ../../nasbench201/dataset/pneumoniamnist.npz --dataset PneumoniaMNIST --size ${s} --batch_size 32 --save_dir ./Penalty15_percentile/ZO_SAP_pneumonia_+${seed}_constraint${s}>pneumonia.log 2>&1
        # nohup python -u ZO-DARTS_SAMVR.py --rand_seed ${seed} --data_path ../../nasbench201/dataset/breastmnist.npz --dataset BreastMNIST --size ${s} --batch_size 16 --save_dir ./Penalty15_percentile/ZO_SAP_breast_+${seed}_constraint${s}>breast.log 2>&1
        # nohup python -u ZO-DARTS_SAMVR.py --rand_seed ${seed} --data_path ../../nasbench201/dataset/tissuemnist.npz --dataset TissueMNIST --size ${s} --batch_size 512 --save_dir ./Penalty15_percentile/ZO_SAP_tissue_+${seed}_constraint${s}>tissue.log 2>&1
        
        nohup python -u ZO-DARTS_SAMVR.py --rand_seed ${seed} --data_path ../../nasbench201/dataset/pathmnist.npz --dataset PathMNIST --size ${s} --batch_size 512 --save_dir ./Penalty15_percentile/ZO_SAP_path_+${seed}_constraint${s}>path.log 2>&1
        nohup python -u ZO-DARTS_SAMVR.py --rand_seed ${seed} --data_path ../../nasbench201/dataset/dermamnist.npz --dataset DermaMNIST --size ${s} --batch_size 128 --save_dir ./Penalty15_percentile/ZO_SAP_derma_+${seed}_constraint${s}>derma.log 2>&1
        nohup python -u ZO-DARTS_SAMVR.py --rand_seed ${seed} --data_path ../../nasbench201/dataset/bloodmnist.npz --dataset BloodMNIST --size ${s} --batch_size 128 --save_dir ./Penalty15_percentile/ZO_SAP_blood_+${seed}_constraint${s}>blood.log 2>&1
    done
done