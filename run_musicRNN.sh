#!/bin/sh

for i in 200 250
do
	for j in 0.001 0.0005
	do
		echo $i $j	
		python music_rnnmlp.py $i $j "./result/weight_saveFile_"$i"_"$j".h5"
	done
done
