#!/bin/bash

# Define the input variables and their values


# Access INPUT SIZE
input_var1_values=("1" "0")

input_var2_values=("2")

input_var3_values=("0.0" "0.1" "64" "128" "256" "512" "1024" "2048" )

# Number of threads
input_var4_values=("0" "1" "2" "36" "72")

# Number of times to run each binary
N=3

# Iterate through combinations of input variables
for var1 in "${input_var1_values[@]}"; do
    for var2 in "${input_var2_values[@]}"; do
        for var3 in "${input_var3_values[@]}"; do
	    for var4 in "${input_var4_values[@]}"; do
                for ((i = 1; i <= N; i++)); do
                    echo "Running with input variables: $var1, $var2, $var3, $var4, (iteration $i)..."
                    ./bin/pad -t "$var4" -a "$var3" -m "$var1" -n "$var2"
		done
            done
        done
    done
done

echo "Script completed."

