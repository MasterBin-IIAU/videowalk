# run model on the dataset
python test.py --filelist eval/davis_vallist.txt --model-type scratch --resume ../pretrained.pth \
--save-path ../results --topk 10 --videoLen 20 --radius 12 --temperature 0.05 \
--cropSize -1 --gpu-id 1

# Convert
python eval/convert_davis.py --in_folder ../results/ --out_folder ../converted_results/ --dataset /data/sdg/tracking_datasets/DAVIS/

# Compute metrics
python /data/sda/v-yanbi/cache/davis2017-evaluation/evaluation_method.py \
--task semi-supervised  --results_path ../converted_results/ --set val \
--davis_path /data/sdg/tracking_datasets/DAVIS/