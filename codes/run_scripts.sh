# single GPU training
python train_cascade.py -opt options/train/ESRGAN.yml
python train.py -opt options/train/DiffGAN.yml
python comparison_figure.py --diff_opt options/train/DiffGAN.yml --clas_name RRDB_DIV_5blocks_interpolated
