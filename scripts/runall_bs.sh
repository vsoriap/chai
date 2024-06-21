#!/bin/bash

# Define the input variables and their values


# Access INPUT SIZE
input_var1_values=("1" "0")

input_var2_values=("16" "32" "64" "128" "256" "512" "1024" "2048" )

# Number of threads
input_var3_values=("0" "1" "2" "36" "72")

input_var4_values=("0" "256")

# Number of times to run each binary
N=3

# Iterate through combinations of input variables
for var1 in "${input_var1_values[@]}"; do
    for var2 in "${input_var2_values[@]}"; do
        for var3 in "${input_var3_values[@]}"; do
	    for var4 in "${input_var4_values[@]}"; do
                for ((i = 1; i <= N; i++)); do
                    echo "Running with input variables: ${id[$var1]}, $var2, $var3, $var4 (iteration $i)..."
                    ./bin/bs -i "$var4" -t "$var3" -l "$var2" -f ${id[$var1]} -c ${od[$var1]}
		done
            done
        done
    done
done

echo "Script completed."

