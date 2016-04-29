//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>

#include <thrust/reduce.h>
#include <thrust/device_vector.h>

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <fstream>

// const struct BlockSize_T
// {
//     int x,y;
// } block_size = {32,32};
const int block_size_x = 32;
const int block_size_y = 32;

enum MaskValues {
    mask_border = 2,
	mask_interior = 1,
	mask_outside = 0
};

#define GLOBALIND(X,Y) ((X) + (Y) * blockDim.x*gridDim.x)
#define GLOBALX(X) ((X) + blockIdx.x * blockDim.x)
#define GLOBALY(Y) ((Y) + blockIdx.y * blockDim.y)
#define IMAGE_IND(X,Y) ((X) + (Y) * image_size_x)
#define ISNOTWHITE(v) ((!((v).x == 255 && (v).y==255 && (v).z==255))&1)

#define MAX(x,y) ((x)>(y) ? (x) : (y))
#define MIN(x,y) ((x)<(y) ? (x) : (y))

typedef union 
{
    uchar4 as_uchar4;
    unsigned char as_array[4];
} uchar4_arr;

typedef struct ChanStruct_T
{
    float* red;
    float* green;
    float* blue;
} ChanStruct;

typedef union ChanStruct_as_array_T
{
    ChanStruct as_chan_struct;
    float* as_array[3];
} ChanStruct_as_array;

struct Box
{
    int x,y,width,height;
    int size_total;
    Box(int x_,int y_,int width_,int height_)
	{
	    x = x_; y=y_; width=width_,height=height_;
	    size_total = width*height;
	}
};

__global__
void prep_data(const uchar4* const d_sourceImg,
	       const uchar4* const d_destImg,
	       int *d_num_neighbors,
	       ChanStruct_as_array sum2_chan_struct,
	       ChanStruct_as_array initial_guess,
	       Box blend_box,
	       int image_size_x,
	       int image_size_y)
{
    //there is an assumption that the image size is an integer multiple of blockDim
    extern __shared__ uchar4 sdata[][block_size_y+2];
    int glob_x = threadIdx.x + blockIdx.x*blockDim.x;
    int glob_y = threadIdx.y + blockIdx.y*blockDim.y;
    int glob_ind = GLOBALIND(glob_x,glob_y);
    
    int sx = threadIdx.x+1, sy = threadIdx.y + 1;

    int image_ind_x = blend_box.x + glob_x;
    int image_ind_y = blend_box.y + glob_y;
    int img_ind = IMAGE_IND(image_ind_x,image_ind_y);
    
    sdata[sx][sy] = d_sourceImg[img_ind];
    
    uchar4_arr my_dest_val;
    my_dest_val.as_uchar4 = d_destImg[img_ind];

    
    const uchar4 white={255,255,255};
    if ( threadIdx.x == 0)
    {
	if ( image_ind_x > 0 )
	{
	    sdata[sx-1][sy] = 
		d_sourceImg[IMAGE_IND(image_ind_x-1,image_ind_y)];
	}
	else
	{
	    sdata[sx-1][sy] = white;
	}
    }
    else if ( threadIdx.x == blockDim.x-1 )
    {
	if ( image_ind_x < image_size_x-1 )
	{
	    sdata[sx+1][sy] = 
		d_sourceImg[IMAGE_IND(image_ind_x+1,image_ind_y)];
	}
	else
	{
	    sdata[sx+1][sy] = white;
	}
    }
    
    if ( threadIdx.y == 0)
    {
	if ( image_ind_y > 0 )
	{
	    sdata[sx][sy-1] = 
		d_sourceImg[IMAGE_IND(image_ind_x,image_ind_y-1)];
	}
	else
	{
	    sdata[sx][sy-1] = white;
	}
    }
    else if ( threadIdx.y == blockDim.y-1 )
    {
	if ( image_ind_y < image_size_y-1 )
	{
	    sdata[sx][sy+1] = 
		d_sourceImg[IMAGE_IND(image_ind_x,image_ind_y+1)];
	}
	else
	{
	    sdata[sx][sy+1] = white;
	}
    }
    __syncthreads();
    
    uchar4_arr my_val;
    my_val.as_uchar4 = sdata[sx][sy];
    int b_my_val = ISNOTWHITE(my_val.as_uchar4);

    uchar4_arr to_left;
    to_left.as_uchar4 = sdata[sx-1][sy];
    int b_to_left = ISNOTWHITE(to_left.as_uchar4);

    uchar4_arr to_right;
    to_right.as_uchar4 = sdata[sx+1][sy];
    int b_to_right = ISNOTWHITE(to_right.as_uchar4);

    uchar4_arr to_top;
    to_top.as_uchar4 = sdata[sx][sy-1];
    int b_to_top = ISNOTWHITE(to_top.as_uchar4);

    uchar4_arr to_bottom;
    to_bottom.as_uchar4 = sdata[sx][sy+1];
    int b_to_bottom = ISNOTWHITE(to_bottom.as_uchar4);

    __syncthreads();
    int my_num_neighbors;
    if ( !b_my_val )
	my_num_neighbors = -1;
    else
    {
	my_num_neighbors = b_to_left
	    + b_to_right
	    + b_to_top
	    + b_to_bottom;
    }
    
    __syncthreads();
    d_num_neighbors[glob_ind] = my_num_neighbors;
    
    for ( int chan_ind = 0 ; chan_ind < 3; chan_ind++ )
    {
	initial_guess.as_array[chan_ind][glob_ind] = my_val.as_array[chan_ind] * ((my_num_neighbors==4)&1) + my_dest_val.as_array[chan_ind] * (1-((my_num_neighbors==4)&1));
    }

    if ( my_num_neighbors == 4 )
    {
	//ChanStruct_as_array sum2_u;
	//sum2_u.as_chan_struct = sum2_chan_struct;
	for ( int chan_ind = 0 ; chan_ind < 3 ; chan_ind++ )
	{
	    float my_sum_value;
	    my_sum_value = 4.0f * my_val.as_array[chan_ind]
		- to_left.as_array[chan_ind]
		- to_right.as_array[chan_ind]
		- to_top.as_array[chan_ind]
		- to_bottom.as_array[chan_ind];
	    sum2_chan_struct.as_array[chan_ind][glob_ind] = my_sum_value;
	}
    }
}

__global__
void single_operation(const float *buff_in,float *buff_out,
		      const float *sum2_chan,
		      const int *num_neighbors)
{
    extern __shared__ float s_Ik[][block_size_y+2];
    int glob_x = threadIdx.x + blockIdx.x*blockDim.x;
    int glob_y = threadIdx.y + blockIdx.y*blockDim.y;
    int glob_ind = GLOBALIND(glob_x,glob_y);

    int my_num_neighbors = num_neighbors[glob_ind];
    if ( my_num_neighbors == -1)
	return;

    int sx = threadIdx.x+1, sy = threadIdx.y + 1;
    int imSizeX = blockDim.x * gridDim.x;
    int imSizeY = blockDim.y * gridDim.y;

    s_Ik[sx][sy] = 
	buff_in[glob_ind];
    
    if ( my_num_neighbors !=4 )
	return;

    if ( threadIdx.x == 0)
    {
	if ( glob_x > 0 )
	{
	    s_Ik[sx-1][sy] = 
		buff_in[GLOBALIND(glob_x-1,glob_y)];
	}
    }
    else if ( threadIdx.x == blockDim.x-1 )
    {
	if ( glob_x < imSizeX-1 )
	{
	    s_Ik[sx+1][sy] = 
		buff_in[GLOBALIND(glob_x+1,glob_y)];
	}
    }
    
    if ( threadIdx.y == 0)
    {
	if ( glob_y > 0 )
	{
	    s_Ik[sx][sy-1] = 
		buff_in[GLOBALIND(glob_x,glob_y-1)];
	}
    }
    else if ( threadIdx.y == blockDim.y-1 )
    {
	if ( glob_y < imSizeY-1 )
	{
	    s_Ik[sx][sy+1] = 
		buff_in[GLOBALIND(glob_x,glob_y+1)];
	}
    }
    __syncthreads();
    
    float my_sum1 = s_Ik[sx-1][sy] + s_Ik[sx+1][sy]
	+ s_Ik[sx][sy-1] + s_Ik[sx][sy+1];
    float my_sum2 = sum2_chan[glob_ind];
    float newVal = (my_sum1 + my_sum2) / 4.f;
    newVal = MIN(255.0f,MAX(newVal,0.0f));
    buff_out[glob_ind] = newVal;
}

__global__
void create_output(uchar4* d_output,
		   ChanStruct_as_array blended,		   
		   const int *num_neighbors,
		   Box blend_box,
		   int image_size_x
    )
{
    int glob_x = threadIdx.x + blockIdx.x*blockDim.x;
    int glob_y = threadIdx.y + blockIdx.y*blockDim.y;
    int glob_ind = GLOBALIND(glob_x,glob_y);

    int my_num_neighbors = num_neighbors[glob_ind];

    if ( my_num_neighbors == 4)
    {
	uchar4 out_value;
	int img_ind_x = blend_box.x + glob_x;
	int img_ind_y = blend_box.y + glob_y;
	int img_ind = img_ind_x + img_ind_y * image_size_x;
	
	out_value.x = blended.as_chan_struct.red[glob_ind];
	out_value.y = blended.as_chan_struct.green[glob_ind];
	out_value.z = blended.as_chan_struct.blue[glob_ind];
	out_value.w=255;
	d_output[img_ind] = out_value;
    }
}

template<typename T> void print_mat(int sizeX,int sizeY,T* d_value,const char* filename)
{
    int sizeTotal = sizeX*sizeY;
    T* h_value = (T*)malloc(sizeTotal*sizeof(T));
    cudaMemcpy(h_value,d_value,sizeof(T)*sizeTotal,
	       cudaMemcpyDeviceToHost);
    std::ofstream f (filename, std::ofstream::out);
    for ( int i = 0 ; i < sizeY ; i++ )
    {
	for ( int j = 0 ; j < sizeX ; j++ )
	{
	    f << h_value[i*sizeX + j];
	    if ( j!=sizeX-1 ) f << ',';
	}
	f << std::endl;
    }
    f.close();

    free(h_value);
}

__global__
void find_non_whites(const uchar4* d_sourceImg,int *x_nonwhite,int *y_nonwhite)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int glob_ind = x + y * gridDim.x * blockDim.x;
    uchar4 my_val = d_sourceImg[glob_ind];
    int non_whiteness = ISNOTWHITE(my_val);
    if ( non_whiteness )
    {
	x_nonwhite[glob_ind] = x;
	y_nonwhite[glob_ind] = y;
    }
    else
    {
	x_nonwhite[glob_ind] = -1;
	y_nonwhite[glob_ind] = -1;
    }
}

template<typename T>
class MinIfnotMinus1
{
public:
    MinIfnotMinus1() {}
    __host__ __device__
    T operator () (T val1,T val2) const
	{
	    if ( val1 == -1 )
		return val2;
	    else if (val2 == -1 )
		return val1;
	    else if (val1<val2)
		return val1;
	    else
		return val2;
	}
};

template<typename T>
class MaxIfnotMinus1
{
public:
    MaxIfnotMinus1() {}
    __host__ __device__
    T operator () (T val1,T val2) const
	{
	    if ( val1 == -1 )
		return val2;
	    else if (val2 == -1 )
		return val1;
	    else if (val1>val2)
		return val1;
	    else
		return val2;
	}
};

Box CalcBox(const uchar4 *d_sourceImg,int sizeX,int sizeY)
{
    int sizeTotal = sizeX*sizeY;
    int *x_nonwhite,*y_nonwhite;
    checkCudaErrors(cudaMalloc(&x_nonwhite,sizeof(int)*sizeTotal));
    checkCudaErrors(cudaMalloc(&y_nonwhite,sizeof(int)*sizeTotal));
    find_non_whites <<< dim3(sizeX/block_size_x,sizeY/block_size_y),
	dim3(block_size_x,block_size_y)>>>
	(d_sourceImg,x_nonwhite,y_nonwhite);
    thrust::device_vector<int> x_vec(x_nonwhite,x_nonwhite+sizeTotal);
    thrust::device_vector<int> y_vec(y_nonwhite,y_nonwhite+sizeTotal);
    
    int x_min = thrust::reduce(x_vec.begin(),x_vec.end(),-1,MinIfnotMinus1<int>());
    int x_max = thrust::reduce(x_vec.begin(),x_vec.end(),-1,MaxIfnotMinus1<int>());
    int y_min = thrust::reduce(y_vec.begin(),y_vec.end(),-1,MinIfnotMinus1<int>());
    int y_max = thrust::reduce(y_vec.begin(),y_vec.end(),-1,MaxIfnotMinus1<int>());
    // std::cout << "x_min " << x_min <<std::endl;
    // std::cout << "x_max " << x_max <<std::endl;
    // std::cout << "y_min " << y_min <<std::endl;
    // std::cout << "y_max " << y_max <<std::endl;

    int size_x = x_max - x_min + 2;
    if ( size_x % block_size_x != 0)
    {
	size_x = (size_x / block_size_x + 1) * block_size_x;
    }
    int size_y = y_max - y_min + 2;
    if ( size_y % block_size_y != 0)
    {
	size_y = (size_y / block_size_y + 1) * block_size_y;
    }
    Box box(x_min-1,y_min-1,size_x,size_y);

    checkCudaErrors(cudaFree(x_nonwhite));
    checkCudaErrors(cudaFree(y_nonwhite));
    return box;
}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{
    int image_size_x,image_size_y,image_size_total;
    cudaStream_t source_init,dest_init;
    checkCudaErrors(cudaStreamCreate(&source_init));
    checkCudaErrors(cudaStreamCreate(&dest_init));

    if ( numRowsSource % block_size_y != 0 )
    {
	image_size_y = (numRowsSource / block_size_y + 1) * block_size_y;
    }
    else
	image_size_y = numRowsSource;

    if ( numColsSource % block_size_x != 0 )
    {
	image_size_x = (numColsSource / block_size_x + 1) * block_size_x;
    }
    else
	image_size_x = numColsSource;
    image_size_total = image_size_x*image_size_y;
        
    uchar4 *d_sourceImg, *d_destImg;
    checkCudaErrors(cudaMalloc(&d_sourceImg,sizeof(uchar4)*image_size_total));
    checkCudaErrors(cudaMemsetAsync((unsigned char*)d_sourceImg,255,sizeof(uchar4)*image_size_total,source_init));
    checkCudaErrors(cudaMemcpy2DAsync(d_sourceImg,image_size_x*sizeof(uchar4),
				      h_sourceImg,numColsSource*sizeof(uchar4),
				      numColsSource*sizeof(uchar4),numRowsSource,
				      cudaMemcpyHostToDevice,source_init));
    
    checkCudaErrors(cudaMalloc(&d_destImg,sizeof(uchar4)*image_size_total));
    checkCudaErrors(cudaMemsetAsync((unsigned char*)d_destImg,255,sizeof(uchar4)*image_size_total,dest_init));
    checkCudaErrors(cudaMemcpy2DAsync(d_destImg,image_size_x*sizeof(uchar4),
				      h_destImg,numColsSource*sizeof(uchar4),
				      numColsSource*sizeof(uchar4),numRowsSource,
				      cudaMemcpyHostToDevice,dest_init));
    
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaStreamDestroy(source_init));
    checkCudaErrors(cudaStreamDestroy(dest_init));
    
    Box blend_box = CalcBox(d_sourceImg,image_size_x,image_size_y);
    int sizeX = blend_box.width;
    int sizeY = blend_box.height;
    int sizeTotal = sizeX*sizeY;
    dim3 grid_size = dim3(sizeX/block_size_x,sizeY/block_size_y);
    
    int* d_num_neighbors;
    checkCudaErrors(cudaMalloc(&d_num_neighbors,sizeof(int) * sizeTotal));
    
    ChanStruct_as_array buffers_structs[2];
    ChanStruct_as_array sum2_chan_struct;
    for ( int chan_ind = 0 ; chan_ind < 3 ; chan_ind++ )
    {
	for ( int i = 0 ; i <2 ; i++ )
	{
	    checkCudaErrors(cudaMalloc(&buffers_structs[i].as_array[chan_ind],sizeof(float)*sizeTotal));
	}
	checkCudaErrors(cudaMalloc(&sum2_chan_struct.as_array[chan_ind],sizeof(float)*sizeTotal));
    }
    
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
     
    prep_data <<< grid_size,dim3(block_size_x,block_size_y),
	(block_size_x+2) * (block_size_y+2) * sizeof(uchar4)>>>
	(d_sourceImg,d_destImg,d_num_neighbors,
	 sum2_chan_struct,
	 buffers_structs[0],
	 blend_box,
	 image_size_x,
	 image_size_y);
    
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
     
    for ( int chan_ind = 0 ; chan_ind < 3 ; chan_ind ++ )
    {
	//initial conditions copied to second buffer
	cudaMemcpy(buffers_structs[1].as_array[chan_ind],
		   buffers_structs[0].as_array[chan_ind],
		   sizeof(float)*sizeTotal,
		   cudaMemcpyDeviceToDevice);
    }
 
    const int num_operations = 800;
    ChanStruct_as_array buffs_in,buffs_out;
     
    cudaStream_t channel_streams[3];
     
    for (int chan_ind = 0 ; chan_ind < 3 ; chan_ind++)
	checkCudaErrors(cudaStreamCreate(&channel_streams[chan_ind]));

    for ( int chan_ind = 0 ; chan_ind < 3 ; chan_ind ++)
    { 
	buffs_in.as_array[chan_ind] = buffers_structs[0].as_array[chan_ind];
	buffs_out.as_array[chan_ind] = buffers_structs[1].as_array[chan_ind];
    }

    for ( int op_ind = 0 ; op_ind < num_operations ; op_ind++ )
    {
	for ( int chan_ind = 0 ; chan_ind < 3 ; chan_ind ++ )
	{
	    single_operation <<<
		grid_size,
		dim3(block_size_x,block_size_y),
		sizeof(float) * (block_size_x+2) * (block_size_y+2),
		channel_streams[chan_ind] >>>
		
		(buffs_in.as_array[chan_ind],buffs_out.as_array[chan_ind],
		 sum2_chan_struct.as_array[chan_ind],
		 d_num_neighbors);
	}
	for ( int chan_ind = 0 ; chan_ind < 3 ; chan_ind ++ )
	{
	    cudaStreamSynchronize(channel_streams[chan_ind]); 
	    checkCudaErrors(cudaGetLastError());
	    float* temp;
	    temp = buffs_in.as_array[chan_ind];
	    buffs_in.as_array[chan_ind] = buffs_out.as_array[chan_ind];
	    buffs_out.as_array[chan_ind] = temp;
	}
    }
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    for (int chan_ind = 0 ; chan_ind < 3 ; chan_ind++)
    {
	checkCudaErrors(cudaStreamDestroy(channel_streams[chan_ind]));
    }

    //print_mat(sizeX,sizeY,sum2_chan_struct.as_array[0],"out.0.txt");
    
    uchar4 *d_output;
    checkCudaErrors(cudaMalloc(&d_output,sizeof(uchar4)*image_size_total));
    checkCudaErrors(cudaMemcpy(d_output,d_destImg,image_size_total*sizeof(uchar4),cudaMemcpyDeviceToDevice));
    create_output <<< grid_size,dim3(block_size_x,block_size_y)>>>
	(d_output,
	 buffs_in,
	 d_num_neighbors,
	 blend_box,
	 image_size_x);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    
    checkCudaErrors(cudaMemcpy2D(h_blendedImg,numColsSource*sizeof(uchar4),
				 d_output,image_size_x*sizeof(uchar4),
				 numColsSource*sizeof(uchar4),numRowsSource,
				      cudaMemcpyDeviceToHost));
    // print_mat(sizeX,sizeY,d_num_neighbors,"num_neighbors.txt");
    // for ( int chan_ind = 0 ; chan_ind < 3 ; chan_ind++ )
    // {
    // 	char s[20];
    // 	sprintf(s,"sum2.%d.txt",chan_ind);
    // 	print_mat(sizeX,sizeY,sum2_chan_struct.as_array[chan_ind],s);
    // }
    checkCudaErrors(cudaFree(d_output));

    for ( int chan_ind = 0 ; chan_ind < 3 ; chan_ind++ )
    {
	for ( int i = 0 ; i <2 ; i++ )
	{
	    checkCudaErrors(cudaFree(buffers_structs[i].as_array[chan_ind]));
	}
	checkCudaErrors(cudaFree(sum2_chan_struct.as_array[chan_ind]));
    }

    cudaFree(d_sourceImg);
    cudaFree(d_destImg);
    cudaFree(d_num_neighbors);
    
    /* To Recap here are the steps you need to implement
  
       1) Compute a mask of the pixels from the source image to be copied
       The pixels that shouldn't be copied are completely white, they
       have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

       2) Compute the interior and border regions of the mask.  An interior
       pixel has all 4 neighbors also inside the mask.  A border pixel is
       in the mask itself, but has at least one neighbor that isn't.

       3) Separate out the incoming image into three separate channels

       4) Create two float(!) buffers for each color channel that will
       act as our guesses.  Initialize them to the respective color
       channel of the source image since that will act as our intial guess.

       5) For each color channel perform the Jacobi iteration described 
       above 800 times.

       6) Create the output image by replacing all the interior pixels
       in the destination image with the result of the Jacobi iterations.
       Just cast the floating point values to unsigned chars since we have
       already made sure to clamp them to the correct range.

       Since this is final assignment we provide little boilerplate code to
       help you.  Notice that all the input/output pointers are HOST pointers.

       You will have to allocate all of your own GPU memory and perform your own
       memcopies to get data in and out of the GPU memory.

       Remember to wrap all of your calls with checkCudaErrors() to catch any
       thing that might go wrong.  After each kernel call do:

       cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

       to catch any errors that happened while executing the kernel.
    */
}
