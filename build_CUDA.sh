#!/bin/bash
echo -e "\033[32mCleaning directory\033[m"
rm -rf bin
mkdir -p bin
module load cuda/12.3 
export CHAI_CUDA_INC=/gpfs/apps/MN5/ACC/CUDA/12.3/include
export CHAI_CUDA_LIB=/gpfs/apps/MN5/ACC/CUDA/12.3/lib64

echo -e "\033[32mCompiling BS version\033[m"
make -C CUDA-U/BS/
cp CUDA-U/BS/bs bin/

echo -e "\033[32mCompiling BFS version\033[m"
make -C CUDA-U/BFS/
cp CUDA-U/BFS/bfs bin/

echo -e "\033[32mCompiling CEDD version\033[m"
make -C CUDA-U/CEDD/
cp CUDA-U/CEDD/cedd bin/

echo -e "\033[32mCompiling CEDT version\033[m"
make -C CUDA-U/CEDT/
cp CUDA-U/CEDT/cedt bin/

echo -e "\033[32mCompiling HSTI version\033[m"
make -C CUDA-U/HSTI/
cp CUDA-U/HSTI/hsti bin/

echo -e "\033[32mCompiling HSTO version\033[m"
make -C CUDA-U/HSTO/
cp CUDA-U/HSTO/hsto bin/

echo -e "\033[32mCompiling PAD version\033[m"
make -C CUDA-U/PAD/
cp CUDA-U/PAD/pad bin/

echo -e "\033[32mCompiling RSCD version\033[m"
make -C CUDA-U/RSCD/
cp CUDA-U/RSCD/rscd bin/

echo -e "\033[32mCompiling RSCT version\033[m"
make -C CUDA-U/RSCT/
cp CUDA-U/RSCT/rsct bin/

echo -e "\033[32mCompiling SC version\033[m"
make -C CUDA-U/SC/
cp CUDA-U/SC/sc bin/

echo -e "\033[32mCompiling SSSP version\033[m"
make -C CUDA-U/SSSP/
cp CUDA-U/SSSP/sssp bin/

echo -e "\033[32mCompiling TRNS version\033[m"
make -C CUDA-U/TRNS/
cp CUDA-U/TRNS/trns bin/


