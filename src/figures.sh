python3 src/generate_angle_generalization_fixed_n.py --checkpoints models/*2.40_indices*/step-000030000.ckpt --metric_json_name step-000030000_metrics_test.json 
python3 src/generate_angle_generalization_fixed_n.py --checkpoints models/*2.40_indices*/step-000030000.ckpt --metric_json_name step-000030000_metrics_test_bounded.json

python3 src/generate_n_metrics.py --checkpoints models/*2.40_indices*/step-000030000.ckpt --metric_json_name step-000030000_metrics_test.json
python3 src/generate_n_metrics.py --checkpoints models/*2.40_indices*/step-000030000.ckpt --metric_json_name step-000030000_metrics_test_bounded.json