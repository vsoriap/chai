#!/bin/bash

# Define the input variables and their values


# Access INPUT SIZE
input_var1_values=("0")

# Number of threads
input_var2_values=("0" "1" "2" "36" "72")

input_var3_values=("0" "256")

id["0"]="input/peppa"

od["0"]="output/peppa"

# Number of times to run each binary
N=3

# Iterate through combinations of input variables
for var1 in "${input_var1_values[@]}"; do
    for var2 in "${input_var2_values[@]}"; do
        for var3 in "${input_var3_values[@]}"; do
            for ((i = 1; i <= N; i++)); do
                echo "Running with input variables: ${id[$var1]}, $var2, $var3 (iteration $i)..."
                ./bin/cedt -i "$var3" -t "$var2" -f ${id[$var1]} -c ${od[$var1]}
            done
        done
    done
done

echo "Script completed."

