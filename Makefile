all: nb2 nb3 nb4

all_cloudless: nb2 wandb ge

nb2:
	python flows/local/rf_flow.py run
	python flows/local/tree_branch_flow.py run
	python flows/local/boosted_flow.py run
	python flows/local/NN_flow.py run

nb3:
	python flows/cloud/rf_flow_cloud.py --environment=conda run --with batch
	python flows/cloud/tree_branch_flow_cloud.py --environment=conda run --with batch

nb4: both_mlops sage_deploy 

both_mlops: wandb ge
	python flows/ecosystem/rf_flow_monitor_validate.py run

wandb: 
	python flows/ecosystem/rf_flow_monitor.py run

ge:
	python flows/ecosystem/iris_validate.py run

sage_deploy: 
	python flows/ecosystem/RF-deploy.py run
