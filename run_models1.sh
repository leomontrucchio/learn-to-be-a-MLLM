export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)

# # VISA
# class_names=("candle" "capsules" "cashew" "chewinggum" "fryum" "macaroni1" "macaroni2" "pcb1" "pcb2" "pcb3" "pcb4" "pipe_fryum")

# # MODEL VISA MAX
# for class_name in "${class_names[@]}"
#     do
#         python3 infer_visa.py --class_name $class_name
#     done
# python3 ./utils_visa/aggregate_metrics_visa.py --label 'grounding_new'



# # LOCO
class_names=("breakfast_box" "juice_bottle" "pushpins" "screw_bag" "splicing_connectors")

# # MODEL LOCO MAX
# for class_name in "${class_names[@]}"
#     do
#         python3 infer_loco_pe3.py --class_name $class_name
#     done
# python3 ./utils_loco/aggregate_metrics_loco.py --label 'hd_generic_0.4'

# MODEL LOCO HD
for class_name in "${class_names[@]}"
    do
        python3 infer_loco_hd.py --class_name $class_name
    done
python3 ./utils_loco/aggregate_metrics_loco.py --label 'hd_generic_0.4'