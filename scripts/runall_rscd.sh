#!/bin/bash

# Define the input variables and their values


# Access INPUT SIZE
input_var1_values=("1" "0")

input_var2_values=("0.0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0" "2.0" )

# Number of threads
input_var3_values=("1" "2" "36" "72")

id["0"]="input/vectors.csv"

od["0"]="output/NYR_bfs_BFS.out"

# Number of times to run each binary
N=3

# Iterate through combinations of input variables
for var1 in "${input_var1_values[@]}"; do
    for var2 in "${input_var2_values[@]}"; do
        for var3 in "${input_var3_values[@]}"; do
            for ((i = 1; i <= N; i++)); do
                echo "Running with input variables: ${id[$var1]}, $var2, $var3, (iteration $i)..."
                ./bin/rscd -t "$var3" -a "$var2" -f ${id[$var1]} 
            done
        done
    done
done

echo "Script completed."

