python main.py --mode train \
                --image_dir data/afhq/train \
                --network_G models.starganv2.model \
                --network_D models.starganv2.model \
                --lambda_sty 0.3 \
                --lambda_ds 1.0 \
                --lambda_cyc 0.1 \
                --beta1 0.0 \
                --beta2 0.99 \
                --g_lr 0.0001 \
                --d_lr 0.0001 \
                --f_lr 0.000001 \
