#!/bin/bash
aihc job submit --pool cce-aycgzcr8 \
                --queue default \
                --name wxl-img2img-turbo-ahic-test \
                --framework 'pytorch' \
                --image ccr-2sgbsvlu-vpc.cnc.bj.baidubce.com/1109backup/aipod:wxl-cuda11.8-torch2_cuda11.1-torch1.9_cuda11.3-torch1.13_1733120674689 \
                --replicas 1 \
                --command "bash /gpfs/public-shared/fileset-groups/wheeljack/wuxiaolei/projs/img2img-turbo/scripts/train_cyclegan.sh" \
                --env CUDA_DEVICE_MAX_CONNECTIONS=1 \
                --gpu baidu.com/a800_80g_cgpu=8 \
                --ds-type pfs \
                --ds-name pfs-ZDtuoX \
                --ds-mountpath /gpfs \
                --priority high#
