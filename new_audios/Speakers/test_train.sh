#80% train
#20% test

rm -R ../test
rm -R ../train
#mkdir ../train
#chmod 777 ../train

if [ ! -d ../test ]; then
	mkdir ../test
	chmod 777 ../test
	mkdir ../train
	chmod 777 ../train
fi

num=0
for x in *; do
	echo $x
	#6 pq el 0 es rename.sh
#	if [ $num == 6 ]; then
#		break
#	fi
	script=$( echo "$x" | grep .sh | wc -l)
	if [ $script != 1 ]; then
		num_files=$(ls "$x" | wc -l)
		#echo $num_files
		train_num=$((num_files*80/100))
		test_num=$(($num_files - $train_num))
		#echo "train $train_num"
		#echo "test $test_num"
		cd $x
		count=0
		for file in *.wav; do
		echo $file
			if [ "$count" -lt "$train_num" ]; then
				cp $file ../../train
			else
				#echo "else"
				#echo $count
				#echo $train_num
				cp $file ../../test
			fi
			#echo $file
			count=$(($count+1))
		done
		cd ..
	fi
	num=$(($num+1))
done
chmod -R 777 ../train
chmod -R 777 ../test
