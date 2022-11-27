result=${PWD##*/}
num=00
for x in *.wav; do
  echo $x
  file_num=$(printf "%03d\n" $num)
  file="${result}_${file_num}.wav"
  echo $file;
  $(mv $x $file)
  num=$(( $num + 1 ))
  echo $x
done

