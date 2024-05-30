!#/bin/bash

#for i in {1000..100000..1000}
#do
	#./or i 0 >> epoch_test_data.txt
#done
gnuplot -e "plot epoch_test_data.txt with lines; pause -1"
