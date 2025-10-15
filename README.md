# gpu-nbody

## Description
A CUDA GPU based solution to parallelize the N-Body simulation problem.

## How to run
1. Copy over the .cu files into your cuda cluster
2. Compile using nvcc
3. Run with one argument, referring to the number of bodies
   3.1. e.g. 
   ```c
   > ./nbody 10000
   ```  
4. Compare the results across different scripts 

