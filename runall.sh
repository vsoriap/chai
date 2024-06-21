#!/bin/bash

# Define the input variables and their values


# Access INPUT SIZE
input_var1_values=("2097152")

# Number of BINS
input_var2_values=("0.0" "1.0" "0.01" "0.05" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" )

# Number of Elements of reduction
input_var3_values=( "512" "1024" "2048" "4096" "8192" "16384" "32768" "65536" "131072" "262144" "524288" "1048576" "2097152" "4194304" "8388608")

# Number of threads
input_var4_values=("0" "1" "2" "3" "6" "8" "12" "16" "32" "36" "64" "72")


# Number of times to run each binary
N=3

# Directory to store output files
output_dir="output_files"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# List of binary executables
binaries=("hsti.atomic" "hsti.atomic-far" "hsti.priv" )

global_output_file="$output_dir/results.csv"

echo "Benchmark,AccessPattern,NumUpdates,NumNops,ReductionSize,NumThreads,AvgExecTime,AvgTroughput,StdExecTime,StdTrhoughput" >$global_output_file

# Iterate through combinations of input variables
for var1 in "${input_var1_values[@]}"; do
    for var2 in "${input_var2_values[@]}"; do
        for var3 in "${input_var3_values[@]}"; do
            for var4 in "${input_var4_values[@]}"; do
                    # Create a unique directory for each combination
                    combo_dir="$output_dir/$var1-$var2-$var3-$var4-$var5"
                    mkdir -p "$combo_dir"
                    
                    # Iterate through each binary
                    for binary in "${binaries[@]}"; do

    			if [[ "$var2" == "0.0" ]] && [[ "$var4" -ne "0" ]]; then
			    break
			fi

                        if [[ "$var2" != "0.0" ]] && [[ "$var4" -eq "0" ]]; then
                            break
                        fi

			if [[ "$binary" == "hsto" ]] && [[ "$var3" -ge "8192" ]] && [[ "$var2" != "1.0" ]]; then
                            break
                        fi

                        if [[ "$binary" == "hsti.priv" ]] && [[ "$var3" -ge "8192" ]] && [[ "$var2" != "1.0" ]]; then
                            break
                        fi

    	                sum_time=0
                        sum_work=0
                        sum_time_squared=0
                        sum_work_squared=0
                        for ((i = 1; i <= N; i++)); do
                            output_file="$combo_dir/$binary-output-$i.txt"
                            echo "Running $binary with input variables: $var1, $var2, $var3, $var4, (iteration $i)..."
                            export OMP_NUM_THREADS="$var5"
                            export CORES="0-$((OMP_NUM_THREADS - 1))"
                
			    if [[ "$var2" == "1.0" ]]; then
				./bin/"$binary" -g 0 -w 1 -r 50 -t "$var4" -b "$var3" -a "$var2" -n "$var1" >> "$output_file" 2>&1
			    else	    
                                ./bin/"$binary" -w 1 -r 50 -t "$var4" -b "$var3" -a "$var2" -n "$var1" >> "$output_file" 2>&1
			    fi    
                            # Extract the value of interest from the output file
                            time=$(cut -d"," -f3 "$output_file")
                                         
                            # Add the result to the sum
                            sum_time=$(bc -l <<< "$sum_time + $time")
                            sum_time_squared=$(bc -l <<< "$sum_time_squared + ($time * $time)")               
                            rm $output_file
                        done
                            
                        # Calculate the average and save it to a separate file
                        average_time=$(bc -l <<< "$sum_time / $N")
                        
                        std_time=$(bc -l <<< "($sum_time_squared / $N) - ($average_time * $average_time)")
                        std_time=$(bc -l <<< "sqrt($std_time)")

                        echo "$average_time,$std_time"
                        echo "$binary,$var1,$var2,$var3,$var4,$average_time,$std_time" >>$global_output_file
                    done
                    rm -rf $combo_dir
            done
        done
    done
done

echo "Script completed."

