for x in *; do
	script=$( echo $x | grep .sh | wc -l)
	echo $script
	if [ $script != 1 ]; then
		cd $x
		cp ../rename_files.sh .;
		chmod 777 "rename_files.sh";
		$(./rename_files.sh)
		rm "rename_files.sh"
		cd ..
	fi
done

