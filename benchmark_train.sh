#!/bin/bash

USERS=2e6
ITEMS=2e5
DESTFILE="/Users/aschioppa/speedTorchData.h5"
BATCH=1024
LATENTFACTORS=128
NUMNEGS=32

python tests/speedTest.py --users $USERS \
        --items $ITEMS \
        --num_latent $LATENTFACTORS \
        --batch_size $BATCH \
        --source_file $DESTFILE \
	--num_negs $NUMNEGS
