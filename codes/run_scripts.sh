# single GPU training
python train.py -opt options/train/DiffGAN.yml
python train.py -opt options/train/ESRGAN.yml
python comparison_figure.py --diff_opt options/train/DiffGAN.yml --clas_opt options/train/ESRGAN.yml 
