#!/bin/bash
curl -L -o nih_data.zip https://www.kaggle.com/api/v1/datasets/download/nih-chest-xrays/data
mkdir -p nih_data
unzip nih_data.zip -d nih_data

rm nih_data.zip