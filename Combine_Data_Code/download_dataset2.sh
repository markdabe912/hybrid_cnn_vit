#!/bin/bash
curl -L -o chexpert_data.zip https://www.kaggle.com/api/v1/datasets/download/ashery/chexpert
mkdir -p chexpert_data
unzip chexpert_data.zip -d chexpert_data

rm chexpert_data.zip