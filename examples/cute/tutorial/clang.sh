clang++ -std=c++17 sgemm_nt_1.cu -I ../../../include -I ../../../tools/util/include/  --cuda-gpu-arch=sm_80    -lcudart_static -ldl -lrt -pthread -O3 -ffast-math -fcuda-flush-denormals-to-zero
