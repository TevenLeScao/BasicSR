# single GPU training
python train.py -opt options/train/ESRGAN.yml
python test.py -opt options/test/ESRGAN.yml > esr_results.txt
#python train.py -opt options/train/NoDiffGAN.yml
#python test.py -opt options/test/NoDiffGAN.yml > msr_results.txt
#python train.py -opt options/train/DiffGAN.yml
#python test.py -opt options/test/DiffGAN.yml > diff_results.txt
