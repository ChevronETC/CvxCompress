#include "CvxCompress_CUDA_kernels.cu"


void CUDA_CvxCompress_InitContainer(
	cudaStream_t stream,
	char* compressed_container,
	int max_num_subvolumes,
	long long max_container_payload_size_in_bytes
	)
{
	dim3 blockShape4(128,1,1);
	dim3 gridShape4(1,1,1);
	cuCvxCompress_InitContainer<<<gridShape4,blockShape4,0,stream>>>(compressed_container,max_num_subvolumes,max_container_payload_size_in_bytes);
}

void CUDA_CvxCompress_CompressSubvolume_To_Container(
    cudaStream_t stream,
    int origin,
    int timestep,
    float* pq0,
	float* pq1,
    float scale,
    char* compressed_sparse,
    char* compressed_container,
    int sv_x0,
    int sv_y0,
    int sv_z0,
    int sv_y1,
    int sv_z1,
    int voi_x0,
    int voi_y0,
    int voi_z0,
    int voi_x1,
    int voi_y1,
    int voi_z1,
	float global_mean,
	float global_rms
    )
{
	int ny = sv_y1 - sv_y0 + 1;
	int nz = sv_z1 - sv_z0 + 1;
	dim3 blockShape2(16,1,16);
	dim3 gridShape2(1,(ny+15)/16,(nz+15)/16);
	cuCvxCompress_CompressSubvolume<<<gridShape2,blockShape2,0,stream>>>(origin,timestep,pq0,pq1,scale,compressed_sparse,sv_x0,sv_y0,sv_z0,sv_y1,sv_z1,voi_x0,voi_y0,voi_z0,voi_x1,voi_y1,voi_z1,global_mean,global_rms);
	dim3 blockShape5(256,1,1);
	dim3 gridShape5(1,1,1);
	cuCvxCompress_AllocNewSubvol<<<gridShape5,blockShape5,0,stream>>>(compressed_container,compressed_sparse);
	dim3 blockShape3(128,1,1);
	int num_copy_gridblocks = ((ny+15)/16) * ((nz+15)/16);
	dim3 gridShape3(num_copy_gridblocks,1,1);
	cuCvxCompress_PackCompressedSparse<<<gridShape3,blockShape3,0,stream>>>(compressed_container,compressed_sparse);
}

void CUDA_CvxCompress_DecompressSubvolume_In_Container(
	cudaStream_t stream,
	char* compressed_container,
	int origin,
	int timestep,
	int sv_x0,
	int sv_y0,
	int sv_z0,
	int sv_y1,
	int sv_z1,
	float* decompressed,
	bool crash_on_error
	)
{
	int ny = sv_y1 - sv_y0 + 1;
	int nz = sv_z1 - sv_z0 + 1;
	dim3 blockShape2(16,1,16);
	dim3 gridShape2(1,(ny+15)/16,(nz+15)/16);
	cuCvxCompress_DecompressSubvolume_In_Container<<<gridShape2,blockShape2,0,stream>>>(compressed_container,origin,timestep,sv_x0,decompressed,false);
}

void CUDA_Copy_To_Buffer(
	cudaStream_t stream,
	const float* vol,
	int nx, int ny, int nz,
	int ny8, int nz8,
	float* d_pq,
	int x0)
{
	// 8 threads for the 8 x-planes, 16x8 tile for y/z
	dim3 block(8, 16, 8);
	dim3 grid(1, (ny8 + 15) / 16, (nz8 + 7) / 8);
	cuCvxCompress_CopyToBuffer<<<grid, block, 0, stream>>>(
		vol, nx, ny, nz, ny8, nz8, d_pq, x0);
}

void CUDA_Copy_From_Buffer(
	cudaStream_t stream,
	float* vol,
	int nx, int ny, int nz,
	int ny8, int nz8,
	const float* d_decomp,
	int x0)
{
	// 16 threads for the 16 x-planes, 8x8 tile for y/z
	dim3 block(16, 8, 8);
	dim3 grid(1, (ny8 + 7) / 8, (nz8 + 7) / 8);
	cuCvxCompress_CopyFromBuffer<<<grid, block, 0, stream>>>(
		vol, nx, ny, nz, ny8, nz8, d_decomp, x0);
}
