# igemm mapping


# from ast import For
# from termios import CKILL
# from turtle import window_height


N = 3
Hi = 4
Wi = 4
C = 8

K = 32
Y = 2
X = 2

pad_h = 0
pad_w = 0
stride_h = 1
stride_w = 1
print(
'''
#               NHWC
# 
#  -----------        -----------
# |           |      |           |
# |           |      |           |
# |           |  K   |           |
# |           |      |           |
# |           |      |           |
#  -----------        -----------
#      YXC              N Ho Wo
# 
#                     -----------                     
#                    |           |                    
#                    |           |                    
#              YXC   |           |                    
#                    |           |                    
#                    |           | 
#                     -----------  
'''
)
# 
# (0, 0, 0, 0) = /sigma (0, Y), (0, X), (0, C) 
Ho = (Hi + 2*pad_h - Y)//stride_h + 1
Wo = (Wi + 2*pad_w - X)//stride_w + 1

gemm_k = Y*X*C
gemm_m = K
gemm_n = N*Ho*Wo

fmt = "{:^3}"

print(f'''
      Output_TENSOR 
      Layout:        NHoWoK
      Dimension:     {N}x{Ho}x{Wo}x{K}
      Tensor Pixels: {N*Ho*Wo*K}
      Expand to im2col Matrix C:
      Axis:          NHoWo x K
      Dimension:     {N*Ho*Wo}x{K}
      Matrix Pixels: {N*Ho*Wo*K}
      ''')
for i_gemm_m in range(gemm_m):
	print('|', end='')
	for i_gemm_n in range(gemm_n):
		i_ho = (i_gemm_n % (Ho*Wo))// Wo
		i_wo = i_gemm_n%Wo

		# print(f'({i_ho},{i_wo})', end ='')
		C_index = i_gemm_n * gemm_m + i_gemm_m
		print(fmt.format(C_index), end='|') if i_gemm_n != gemm_n-1 else print(fmt.format(C_index), '|')

print(f'''
      Input_TENSOR 
      Layout:        NHiWiC
      Dimension:     {N}x{Hi}x{Wi}x{C}
      Tensor Pixels: {N*Hi*Wi*C}
      Expand to im2col Matrix B:
      Axis:          NHoWo x YXC
      Dimension:     {N*Ho*Wo}x{Y*X*C}
      Matrix Pixels: {N*Ho*Wo*Y*X*C}
      ''')
print("                                     'x' means out of bound and equal to zero")
for i_gemm_k in range(gemm_k):
	if i_gemm_k % (C) == 0:
		print('-'+'----'*gemm_n)
	print('|', end='')
	for i_gemm_n in range(gemm_n):
		i_n = i_gemm_n // (Ho*Wo)
		i_ho = (i_gemm_n % (Ho*Wo))// Wo
		i_wo = i_gemm_n%Wo


        # YXC
		i_y = i_gemm_k // (X*C)
		i_x = (i_gemm_k % (X*C))//C
		i_c = i_gemm_k % C

		# CYX
		# i_c = i_gemm_k // (Y*X)
		# i_y = (i_gemm_k % (Y*X))//X
		# i_x = i_gemm_k % X

		i_hi = (i_ho - 1) * stride_h + i_y - 2*pad_h
		i_wi = (i_wo - 1) * stride_w + i_x - 2*pad_w
		# print(f'({i_hi},{i_wi})', end ='')

		if i_hi<0 or i_wi<0:
			input_index = 'x'
		else:
			input_index = i_n * Hi * Wi * C + i_hi * Wi * C + i_wi * C + i_c

		print(fmt.format(input_index), end = '|') if i_gemm_n != gemm_n-1 else print(fmt.format(input_index), end='|\n')

print(f'''
      Weight_TENSOR 
      Layout:        KCYXc
      Dimension:     {K}x{Y}x{X}x{C}
      Tensor Pixels: {K*Y*X*C}
      Expand to im2col Matrix A:
      Axis:          K x YXC
      Dimension:     {K}x{Y*X*C}
      Matrix Pixels: {K*Y*X*C}
      ''')

for i_gemm_m in range(gemm_m):
    # print('-'+'----'*gemm_k)
    for i_gemm_k in range(gemm_k):
        if i_gemm_k % C ==0:
            print('|', end='')
        weight_index = i_gemm_m * gemm_k + i_gemm_k
        print(fmt.format(weight_index), end = ' ') if i_gemm_k != gemm_k-1 else print(fmt.format(weight_index), end='|\n')


Vec = 8
print(
f'''
#               NCHWC(Vec={Vec})
# 
#  -----------        -----------
# |           |      |           |
# |           |      |           |
# |           |  KVec|           |
# |           |      |           |
# |           |      |           |
#  -----------        -----------
#      CYXVec            N Ho Wo
# 
#                     -----------                     
#                    |           |                    
#                    |           |                    
#              CYXVec|           |                    
#                    |           |                    
#                    |           | 
#                     -----------  
'''
)
# 
# (0, 0, 0, 0) = /sigma (0, Y), (0, X), (0, C) 
Ho = (Hi + 2*pad_h - Y)//stride_h + 1
Wo = (Wi + 2*pad_w - X)//stride_w + 1

gemm_k = C*Y*X
gemm_m = K
gemm_n = N*Ho*Wo

C_vec = C//Vec
K_vec = K//Vec

fmt = "{:^3}"

print(f'''
      Output_TENSOR 
      Layout:        NKHoWok
      Dimension:     {N}x{K_vec}x{Ho}x{Wo}x{Vec}
      Tensor Pixels: {N*K_vec*Ho*Wo*Vec}
      Expand to im2col Matrix C:
      Axis:          Kk x NHoWo
      Dimension:     {K_vec*Vec}x{N*Ho*Wo}
      Matrix Pixels: {N*K_vec*Ho*Wo*Vec}
      ''')
for i_gemm_m in range(gemm_m):
    if i_gemm_m % (Vec) == 0:
        print('-'+'----'*gemm_n)
    print('|', end='')
    for i_gemm_n in range(gemm_n):
        i_n= i_gemm_n // (Ho*Wo)
        i_ho = (i_gemm_n % (Ho*Wo))// Wo
        i_wo = i_gemm_n%Wo
        i_k = i_gemm_m // Vec
        i_vec = i_gemm_m % Vec

		# print(f'({i_ho},{i_wo})', end ='')
        C_index = i_n * K_vec * Ho * Wo * Vec + i_k * Ho * Wo * Vec + i_ho * Wo * Vec + i_wo * Vec + i_vec
        print(fmt.format(C_index), end='|') if i_gemm_n != gemm_n-1 else print(fmt.format(C_index), '|')

print(f'''
      Input_TENSOR 
      Layout:        NCHiWic
      Dimension:     {N}x{C_vec}x{Hi}x{Wi}x{Vec}
      Tensor Pixels: {N*C_vec*Hi*Wi*Vec}
      Expand to im2col Matrix B:
      Axis:          NHoWo x CYXc
      Dimension:     {N*Ho*Wo}x{C_vec*Y*X*Vec}
      Matrix Pixels: {N*Ho*Wo*C_vec*Y*X*Vec}
      ''')
print("                                     'x' means out of bound and equal to zero")
for i_gemm_k in range(gemm_k):
	if i_gemm_k % (Vec) == 0:
		print('-'+'----'*gemm_n)
	print('|', end='')
	for i_gemm_n in range(gemm_n):
		i_n = i_gemm_n // (Ho*Wo)
		i_ho = (i_gemm_n % (Ho*Wo))// Wo
		i_wo = i_gemm_n%Wo


        # CYXvec
		i_c = i_gemm_k // (Y*X*Vec)
		i_y = (i_gemm_k % (Y*X*Vec))//(X*Vec)
		i_x = ((i_gemm_k % (Y*X*Vec))%(X*Vec))//Vec
		i_vec = i_gemm_k % Vec

		# CYX
		# i_c = i_gemm_k // (Y*X)
		# i_y = (i_gemm_k % (Y*X))//X
		# i_x = i_gemm_k % X

		i_hi = (i_ho - 1) * stride_h + i_y - 2*pad_h
		i_wi = (i_wo - 1) * stride_w + i_x - 2*pad_w
		# print(f'({i_hi},{i_wi})', end ='')

		if i_hi<0 or i_wi<0:
			input_index = 'x'
		else:
			input_index = i_n * C_vec * Hi * Wi * Vec + i_c* Hi * Wi * Vec + i_hi * Wi * Vec + i_wi * Vec + i_vec

		print(fmt.format(input_index), end = '|') if i_gemm_n != gemm_n-1 else print(fmt.format(input_index), end='|\n')

print(f'''
      Weight_TENSOR 
      Layout:        KCYXc
      Dimension:     {K}x{C_vec}x{Y}x{X}x{Vec}
      Tensor Pixels: {K*C_vec*Y*X*Vec}
      Expand to im2col Matrix A:
      Axis:          K x CYXc
      Dimension:     {K}x{C_vec*Y*X*Vec}
      Matrix Pixels: {K*C_vec*Y*X*Vec}
      ''')

for i_gemm_m in range(gemm_m):
    # print('-'+'----'*gemm_k + '-'*(gemm_k//5))
    for i_gemm_k in range(gemm_k):
        if i_gemm_k % Vec ==0:
            print('|', end='')
        weight_index = i_gemm_m * gemm_k + i_gemm_k
        print(fmt.format(weight_index), end = ' ') if i_gemm_k != gemm_k-1 else print(fmt.format(weight_index), end='|\n')