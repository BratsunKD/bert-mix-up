#!/bin/bash

# Запуск скрипта обучения
python3 lib/train.py --mixup_type $1

#python3 lib/train.py --mixup_type none 
#python3 lib/train.py --mixup_type embedding
#python3 lib/train.py --mixup_type sentences

