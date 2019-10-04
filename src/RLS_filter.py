import numpy as np
import math

def generate_local_difference(data, n):
    # To remove spatial Coorelation list

    '''
    1. Convert 2D Matrix to 1D Matrix
    2. Calculate sum of n*(n+1)*2 elements for each element of 1D matrix.  
        s'(t) = s(t-1) + s(t-2) + s(t-3) + .......... + s(t-n)
        S1 = sum( s(t-1 -k*W) + s(t-2 - k*W) + s(t-3 -k*W) + .......... + s(t-n -k*W) for k = 1 to n
        S2 = sum( s(t+1 -k*W) + s(t+2 - k*W) + s(t+3 -k*W) + .......... + s(t+n -k*W) for k = 1 to n
        S0 = sum( s(t -k*W) ) for k = 1 to n
        s'(t) = s'(t) + S1 + S2 + S0
    3. Append the result in the local_difference list

    Reference link:

    '''

    # 1. Convert 2D to 1D
    W, H = data.shape
    print("\tShape of dataset within a band"+ str(data.shape))
    data = data.flatten()
    print("\tShape of dataset within a band after flatten"+ str(data.shape))

    # 2. Fill local difference
    local_difference = []
    length = len(data)

    for i in range(length):
        s_ = 0
        j = 1
        while( i-j >=0 and j <=n ):
            s_ = s_ + data[i-j]
            j = j + 1
          
        s1 = 0
        for k in range(n):
            j = 1
            while( i-j -(k+1)*W >= 0 and j<=n):
                s1 = s1 + data[i-j -(k+1)*W]
                j = j + 1

        s2 = 0
        for k in range(n):
            j = 1
            while( i-j -(k+1)*W >= 0 and j<=n):
                s2 = s2 + data[i+j -(k+1)*W]
                j = j + 1

        s0 = 0
        k = 1
        while( i - k*W >=0 and k<=n ):
            s0 = s0 + data[i - k*W]
            k = k + 1    

        s_ = s_ + s0 + s1 + s2
        s_ = s_/(n*(n+1)*2)

        local_difference.append(s_ - s0)
    
    return local_difference

def get_residual_error_by_band(local_diff_matrix, p, z):
    '''
    local_diff_matrix size( W*H, 1, bands)
    z: band number
    p: number of bands used for predicting zth band
    t: 0 to W*H-1 
    T: Transpose

    # Step 1 Declaration
    input_vector: dz(t) = [dz-1(t), dz-2(t), dz-3(t), ...... dz-p(t)]
    size of input vector: (1,p)
    Weight_vector: W(t) [W1(t), W2(t), W3(t), ..... Wp(t)]
    size of Wg_vector: (1,p)

    # Step 2 Initialization
    t = 1
    delta = 0.0001
    p(0) = delta*I , where I is Identitiy_matrix (pXp)
    calculate : dz-i(t) from i = 1 to p
    then form the input vector dz(t)

    # Step 3 calculate prediction residual 
    ez(t) = dz(t) - floor(dz(t).w(t-1)^T)
    size of ez(t): (1,p)
    K^T(t) = p(t-1).dz^T(t)/( 1 + dz(t).p(t-1).dz^T(t) )
    size of K^T(t): (p,1)
    p(t) = p(t-1) - K^T(t).dz(t).p(t-1)
    size of p(t): (p,p)

    # Step 4 Update Weight_Matrix
    w(t) = w(t-1) + K(t).ez(t)

    # Step 5 Send the value of ez(t) to the encoder

    # Step 6 
    t = t+1size of K^T(t): (p,1)
    go to step 2 until t <= W*H

    # Step 7
    z = z+1
    go to step 1 until z <= Z
    
    '''

    if(z <= p):
        print("\t\tp is greater than z, add appropriate p")
        p = z-1

    #local_diff_matrix = np.array(local_diff_matrix)
    shape = local_diff_matrix.shape
    t = 1
    delta = 0.0001

    p_Delta = delta*np.eye(p)
    print("\t\tshape of p_Delta: " + str(p_Delta.shape))
    wt_matrix = np.zeros((1,p))
    print("\t\tshape of wt_matrix: " + str(wt_matrix.shape))

    ez_array = []

    while( t <= shape[0]):
        print("\t\tt " + str(t) )
        dz = local_diff_matrix[t-1, z-p-1:z-1]
        dz = dz.reshape((1,p))
        print("\t\tshape of dz: " + str(dz.shape))
        
        
        dz_Transpose = np.reshape(dz, (p,1))

        # Step 3 calculate prediction residual 
        wt_matrix_Transpose = wt_matrix.reshape((p,1))

        ez = local_diff_matrix[t-1, z-1] - math.floor( np.dot(dz, wt_matrix_Transpose) ) 

        K_coeff =  1 + np.dot(dz, np.dot( p_Delta, dz_Transpose))
        K = np.dot( p_Delta, dz_Transpose)
        K = K/K_coeff

        K_transpose = np.reshape(K, (1,p))
        p_Delta = p_Delta - np.dot( K, np.dot( dz, p_Delta) )
        

        # Step 4 Update Weight_Matrix
    
        wt_matrix = wt_matrix + K_transpose*ez
        print("\t\tvalue of local differnece: " + str(local_diff_matrix[t-1, z-1]) )
        print("\t\tvalue of ez: " + str(ez))

        # Step 5 
        ez_array.append(ez)
        
        # Step 6
        t = t+1
        
    return ez_array


def get_residual_error(local_diff_matrix, p):
    shape = local_diff_matrix.shape
    Z = shape[1]

    ez_matrix = []
    print("no of bands " + str(Z))

    for z in range(Z):
        print("\n\tband no " + str(z))

        if(z == 0):   # 1st band
            ez_array = local_diff_matrix[:, 0]
        else:
            ez_array = get_residual_error_by_band(local_diff_matrix, p, z+1)
        
        print("\tlen of ez_Array " + str(len(ez_array)) )

        ez_matrix.append(ez_array)
        print("\tlen of ez_matrix " + str(len(ez_matrix)) )
    
    ez_matrix = np.array(ez_matrix)
    return ez_matrix
