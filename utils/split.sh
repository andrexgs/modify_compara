#!/bin/bash

#
#    Script - split data between train and test 
#
#    Name: split.sh
#    Author: Gabriel Kirsten Menezes (gabriel.kirsten@hotmail.com)
#            Hemerson Pistori
#

echo "[SCRIPT SPLIT DATA] Initializing..."

dir_all="../data/all"
dir_train="../data/train"
dir_test="../data/test"

mkdir -p $dir_train
mkdir -p $dir_test

rm -rf $dir_train/*
rm -rf $dir_test/*

echo 'Copy from ' $dir_all ' to '$dir_train

cp -R $dir_all/* $dir_train

# Precisa passar o percentual de treinamento como parâmetro
perc_train=$1
max_train=84  # All classes will have only max_train images for training
              # Use -1 for not balancing
              # Passe 100 como parâmetro para o percentual

for dir_class in `ls $dir_train`;
do
    echo "[SCRIPT SPLIT DATA] Spliting class -" $dir_class;
    mkdir $dir_test/$dir_class
    quantity_files=`ls $dir_train/$dir_class | wc -l`

    perc_quantity_files_float=`echo "scale=2; ($quantity_files/100)*$perc_train" | bc -l `

    perc_quantity_files=${perc_quantity_files_float%.*}

    # Move 100-perc_train
    counter=0
    arrayFiles=`ls $dir_train/$dir_class |sort -R`
    for file in $arrayFiles;
    do
        let "counter += 1"
        if [[ $counter -gt $perc_quantity_files ]]; then
            mv $dir_train/$dir_class/$file $dir_test/$dir_class/$file
        fi
    done


    if [[ $max_train -gt 0 ]]; then
       counter=0
       arrayFiles=`ls $dir_train/$dir_class |sort -R`
       for file in $arrayFiles;
       do
          let "counter += 1"
          if [[ $counter -gt $max_train ]]; then
              rm $dir_train/$dir_class/$file
          fi
       done
    fi
done

echo "[SCRIPT SPLIT DATA] OK! DONE."
