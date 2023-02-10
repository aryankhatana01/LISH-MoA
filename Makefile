create_folds:
	python3 create_folds.py

run_training:
	cd src && python3 train.py

format_code:
	black .

run_inference:
	cd inference && python3 inference.py
