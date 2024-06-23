#!/bin/bash

mkdir -p submit

python3 lib/inference.py --mixup_type $1

#python3 lib/inference.py --mixup_type none
#python3 lib/inference.py --mixup_type embedding
#python3 lib/inference.py --mixup_type sentences
