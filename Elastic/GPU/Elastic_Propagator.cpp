#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <xmmintrin.h>
#include <cuda_runtime_api.h>

#include "gpuAssert.h"
#include "Elastic_Buffer.hxx"
#include "Elastic_Pipeline.hxx"
#include "Elastic_Propagator.hxx"
#include "Elastic_Modeling_Job.hxx"
#include "Elastic_Shot.hxx"
#include "Global_Coordinate_System.hxx"
#include "Voxet.hxx"

Elastic_Propagator::Elastic_Propagator(Elastic_Modeling_Job* job)
{
	job->_propagator = this;
	_init(
			job->Get_Log_Level(),
			job,
			job->Get_Propagation_NX(),
			job->Get_Propagation_NY(),
			job->Get_Propagation_NZ(),
			job->Get_DX(),
			job->Get_DY(),
			job->Get_DZ(),
			8,
			job->Get_Number_Of_GPU_Pipes(),
			job->Get_Steps_Per_GPU(),
			job->Get_GPU_Devices(),
			job->Get_Number_Of_GPU_Devices(),
			false
	     );
}

void Elastic_Propagator::_init(
	int log_level,
	Elastic_Modeling_Job* job,
	int nx, 
	int ny, 
	int nz,
	float dx,
	float dy,
	float dz,
	int Stencil_Order, 
	int num_pipes, 
	int num_timesteps, 
	const int* device_id, 
	int num_devices,
	bool debug
	)
{
	_log_level = log_level;
	_dti = 0.0;
	_num_timesteps = 0;
	_job = job;
	_debug = debug;
	_num_devices = num_devices;
	_device_id = new int[_num_devices];
	_cmp_streams = new cudaStream_t[_num_devices];
	_inp_streams = new cudaStream_t[_num_devices];
	_out_streams = new cudaStream_t[_num_devices];
	for (int i = 0;  i < _num_devices;  ++i)
	{
		_device_id[i] = device_id[i];
		_cmp_streams[i] = 0L;
		_inp_streams[i] = 0L;
		_out_streams[i] = 0L;
	}

	_stencil_order = Stencil_Order;

	_dx = dx;
	_dy = dy;
	_dz = dz;

	_nx = nx;
	_ny = ny;
	_nz = nz;

	_bsX = Stencil_Order / 2;
	_NbX = (_nx + _bsX - 1) / _bsX;

	_ts = 0L;
	_PV = 0L;
	_ST = 0L;
	_EM = 0L;
	_pbuf_PV = 0L;
	_pbuf_ST = 0L;
	_pbuf_EM = 0L;
	_pbuf_PV_Out = 0L;
	_pbuf_ST_Out = 0L;
	_pbuf_EM_Out = 0L;

	_pbuf_first_call = true;

	_tried_p2p = new bool*[num_devices];
	for (int i = 0;  i < num_devices;  ++i)
	{
		_tried_p2p[i] = new bool[num_devices];
		for (int j = 0;  j < num_devices;  ++j)
		{
			_tried_p2p[i][j] = (i == j) ? true : false;
		}
	}

	// ..determine relative cost of two kernels. For now, this is just hard-coded
	_rel_cost = new double[2];
	_rel_cost[0] = 0.5;  // T kernel
	_rel_cost[1] = 0.5;  // V kernel

	if (Check_GPUs(_device_id, num_devices))
	{
		_num_pipes = num_pipes;
		_GPUs_per_pipe = num_devices / num_pipes;
		int half_stencil = Stencil_Order / 2;
		int z0 = 0;
		int z1 = nz-1;

		// load balancing
		int* pipe_width = new int[num_pipes];

		// ..between pipes
		if (num_pipes > 2)
		{
			double half_halo_cost = 0.0;
			for (int i = 1;  i < _GPUs_per_pipe*num_timesteps*2;  ++i) half_halo_cost += (double)(i * half_stencil * _rel_cost[i&1]);
			double extra_work = half_halo_cost / (double)(_GPUs_per_pipe * num_timesteps * num_pipes);
			extra_work = 2.0 * extra_work / (_rel_cost[0] + _rel_cost[1]);
			double y0 = 0.0;
			for (int iPipe = 0;  iPipe < num_pipes;  ++iPipe)
			{
				double y1 = y0 + ((double)ny/(double)num_pipes) + ((iPipe == 0 || iPipe == (num_pipes-1)) ? extra_work * (double)(num_pipes - 2) / 2.0 : -extra_work);
				pipe_width[iPipe] = (int)round(y1) - (int)round(y0);
				y0 = y1;
			}

			double* cost = new double[num_pipes];
			double total_cost = 0.0;
			for (int iPipe = 0, y0 = 0;  iPipe < num_pipes;  y0+=pipe_width[iPipe++]) 
			{
				cost[iPipe] = Calculate_Cost(y0,pipe_width[iPipe],ny,num_timesteps,_GPUs_per_pipe,half_stencil,_rel_cost);
				total_cost += cost[iPipe];
			}
			printf("Load balancing between pipes ::\n");
			for (int iPipe = 0, y0=0;  iPipe < num_pipes;  y0+=pipe_width[iPipe++])
			{
				printf("  Pipe %d :: y=[%d,%d], width=%d, cost=%.2f%%\n",iPipe+1,y0,y0+pipe_width[iPipe]-1,pipe_width[iPipe],100.0*cost[iPipe]/total_cost);
			}
			delete [] cost;
		}
		else
		{
			int k = (ny + num_pipes - 1) / num_pipes;
			for (int iPipe = 0;  iPipe < num_pipes;  ++iPipe) pipe_width[iPipe] = k;
			pipe_width[num_pipes-1] -= k*num_pipes - ny;
		}

		// ..between gpus within each pipe
		_pipes = new Elastic_Pipeline*[num_pipes];
		for (int iPipe = 0, pipe_y0 = 0;  iPipe < num_pipes;  pipe_y0+=pipe_width[iPipe++])
		{
			int** steps = new int*[_GPUs_per_pipe*num_timesteps*2];
			int y0 = pipe_y0;
			int y1 = y0 + pipe_width[iPipe] - 1;
			_pipes[iPipe] = new Elastic_Pipeline(this,iPipe,y0,y1,0,nz-1);
			double total_cost = 0.0;
			for (int i = _GPUs_per_pipe*num_timesteps*2-1;  i >= 0;  --i)
			{
				steps[i] = new int[3];
				total_cost += (double)(y1 - y0 + 1) * _rel_cost[i&1];

				steps[i][0] = y0;
				steps[i][1] = y1;

				// next iteration
				y0 -= half_stencil;
				y1 += half_stencil;
				if (y0 < 0) y0 = 0;
				if (y1 >= ny) y1 = ny-1;
			}

			double max_cost_per_GPU = total_cost / (double)_GPUs_per_pipe;
			//printf("\nLoad balancing within pipe %d\n",iPipe+1);
			double cost = 0.0;
			int curr_device_id = -1, cbo = 0, emcbo = 0;
			int min_y = 1000000000, max_y = -1000000000, curr_steps = 0;
			Elastic_Buffer* prev_EM = 0L;
			for (int i = 0, iGPU=0, iStep=0;  i < _GPUs_per_pipe*num_timesteps*2;  ++i)
			{
				int* vals = steps[i];
				int ylen = vals[1] - vals[0] + 1;
				if (i == 0)
				{
					//printf("..GPU %d\n",iGPU+1);
					curr_device_id = device_id[iPipe*_GPUs_per_pipe+iGPU];

					Elastic_Buffer* tmp = new Elastic_Buffer(this,_pipes[iPipe],curr_device_id,iStep/2,(iStep&1)==1,vals[0],vals[1],z0,z1,3,cbo,0L,0);
					_pipes[iPipe]->Append_Buffer(tmp);

					int inp_b_y0 = vals[0] - half_stencil;
					int inp_b_y1 = vals[1] + half_stencil;
					if (inp_b_y0 < 0) inp_b_y0 = 0;
					if (inp_b_y1 >= ny) inp_b_y1 = ny-1;
					tmp = new Elastic_Buffer(this,_pipes[iPipe],curr_device_id,(iStep+1)/2,((iStep+1)&1)==1,inp_b_y0,inp_b_y1,z0,z1,4,cbo,0L,0);
					_pipes[iPipe]->Append_Buffer(tmp);

					emcbo = cbo;
					cbo -= 2;
				}
				double local_cost = (double)ylen * _rel_cost[i&1];
				if (cost + local_cost >= max_cost_per_GPU)
				{
					if (cost + local_cost > max_cost_per_GPU)
					{
						vals[2] = (int)round((double)ylen * (max_cost_per_GPU - cost) / local_cost);

						// partial step - sending end
						double split_cost = local_cost * (double)vals[2] / (double)ylen;
						cost = cost + split_cost;
						Elastic_Buffer* tmp = new Elastic_Buffer(
								this,_pipes[iPipe],curr_device_id,(iStep+2)/2,((iStep+2)&1)==1,vals[0],vals[0]+vals[2]-1,vals[0],vals[0]+vals[2]-1,z0,z1,2,cbo,0,
								_pipes[iPipe]->Get_Buffer(-2),_pipes[iPipe]->Get_Buffer(-1),0L,0
								);
						--cbo;
						++curr_steps;
						tmp->Add_To_YRange(min_y,max_y);
						_pipes[iPipe]->Append_Buffer(tmp);

						prev_EM = new Elastic_Buffer(this,_pipes[iPipe],curr_device_id,min_y,max_y,z0,z1,curr_steps+2,emcbo,prev_EM,0);
						_pipes[iPipe]->Add_EM_Buffer(prev_EM);
						curr_steps = 0;
						min_y = 1000000000;
						max_y = -1000000000;

						++iGPU;
						if (iGPU >= _GPUs_per_pipe) break;
						cost = local_cost - split_cost;
						//printf("..GPU %d\n",iGPU+1);
						curr_device_id = device_id[iPipe*_GPUs_per_pipe+iGPU];

						// partial step - receiving end
						tmp = new Elastic_Buffer(this,_pipes[iPipe],curr_device_id,iStep/2,(iStep&1)==1,vals[0]+vals[2],vals[1],z0,z1,3,cbo+1,_pipes[iPipe]->Get_Buffer(-4),0);
						_pipes[iPipe]->Append_Buffer(tmp);

						int inp_b_y0 = vals[0] - half_stencil;
						int inp_b_y1 = vals[1] + half_stencil;
						if (inp_b_y0 < 0) inp_b_y0 = 0;
						if (inp_b_y1 >= ny) inp_b_y1 = ny-1;
						tmp = new Elastic_Buffer(this,_pipes[iPipe],curr_device_id,(iStep+1)/2,((iStep+1)&1)==1,inp_b_y0,inp_b_y1,z0,z1,4,cbo+1,_pipes[iPipe]->Get_Buffer(-4),0);
						_pipes[iPipe]->Append_Buffer(tmp);

						tmp = new Elastic_Buffer(
								this,_pipes[iPipe],curr_device_id,(iStep+2)/2,((iStep+2)&1)==1,vals[0],vals[1],vals[0]+vals[2],vals[1],z0,z1,4,cbo,1,
								_pipes[iPipe]->Get_Buffer(-2),_pipes[iPipe]->Get_Buffer(-1),_pipes[iPipe]->Get_Buffer(-4),0
								);
						emcbo = cbo + 1;
						cbo -= 2;
						++curr_steps;
						tmp->Add_To_YRange(min_y,max_y);
						_pipes[iPipe]->Append_Buffer(tmp);
					}
					else
					{
						vals[2] = 0;

						// full step - sending end
						Elastic_Buffer* tmp = new Elastic_Buffer(
								this,_pipes[iPipe],curr_device_id,(iStep+2)/2,((iStep+2)&1)==1,vals[0],vals[1],vals[0],vals[1],z0,z1,2,cbo,0,
								_pipes[iPipe]->Get_Buffer(-2),_pipes[iPipe]->Get_Buffer(-1),0L,0
								);
						--cbo;
						tmp->Add_To_YRange(min_y,max_y);
						++curr_steps;
						_pipes[iPipe]->Append_Buffer(tmp);

						prev_EM = new Elastic_Buffer(this,_pipes[iPipe],curr_device_id,min_y,max_y,z0,z1,curr_steps+3,emcbo,prev_EM,0);
						_pipes[iPipe]->Add_EM_Buffer(prev_EM);
						emcbo = cbo;
						curr_steps = 0;
						min_y = 1000000000;
						max_y = -1000000000;

						++iGPU;
						if (iGPU >= _GPUs_per_pipe) break;

						cost = cost + local_cost - max_cost_per_GPU;
						//printf("..GPU %d\n",iGPU+1);
						curr_device_id = device_id[iPipe*_GPUs_per_pipe+iGPU];

						// full step - receiving end
						tmp = new Elastic_Buffer(this,_pipes[iPipe],curr_device_id,(iStep+1)/2,((iStep+1)&1)==1,vals[0],vals[1],z0,z1,3,cbo,_pipes[iPipe]->Get_Buffer(-3),0);
						_pipes[iPipe]->Append_Buffer(tmp);

						int inp_b_y0 = vals[0] - half_stencil;
						int inp_b_y1 = vals[1] + half_stencil;
						if (inp_b_y0 < 0) inp_b_y0 = 0;
						if (inp_b_y1 >= ny) inp_b_y1 = ny-1;
						tmp = new Elastic_Buffer(this,_pipes[iPipe],curr_device_id,(iStep+2)/2,((iStep+2)&1)==1,inp_b_y0,inp_b_y1,z0,z1,4,cbo,_pipes[iPipe]->Get_Buffer(-3),0);
						_pipes[iPipe]->Append_Buffer(tmp);
						cbo -= 2;
					}
				}
				else
				{
					vals[2] = 0;
					cost = cost + local_cost;
					Elastic_Buffer* tmp = new Elastic_Buffer(
							this,_pipes[iPipe],curr_device_id,(iStep+2)/2,((iStep+2)&1)==1,vals[0],vals[1],vals[0],vals[1],z0,z1,3,cbo,0,
							_pipes[iPipe]->Get_Buffer(-2),_pipes[iPipe]->Get_Buffer(-1),0L,0
							);
					--cbo;
					++curr_steps;
					tmp->Add_To_YRange(min_y,max_y);
					_pipes[iPipe]->Append_Buffer(tmp);
				}
				++iStep;
			}
			_pipes[iPipe]->Get_Buffer(-3)->Set_Is_Device2Host(true);
			_pipes[iPipe]->Get_Buffer(-2)->Set_Is_Device2Host(true);
			_pipes[iPipe]->Get_Buffer(-1)->Set_Is_Device2Host(_debug);
		}
	
		delete [] pipe_width;
	}
}

Elastic_Propagator::~Elastic_Propagator()
{
	if (_tried_p2p != 0L)
	{
		for (int i = 0;  i < _num_devices;  ++i)
		{
			if (_tried_p2p[i] != 0L) delete [] _tried_p2p[i];
		}
		delete [] _tried_p2p;
		_tried_p2p = 0L;
	}
	for (int i = 0;  i < _num_devices;  ++i)
	{
		if (_cmp_streams[i] != 0L)
		{
			cudaStreamDestroy(_cmp_streams[i]);
			_cmp_streams[i] = 0L;
		}
		if (_inp_streams[i] != 0L)
		{
			cudaStreamDestroy(_inp_streams[i]);
			_inp_streams[i] = 0L;
		}
		if (_out_streams[i] != 0L)
		{
			cudaStreamDestroy(_out_streams[i]);
			_out_streams[i] = 0L;
		}
	}
	delete [] _cmp_streams;
	delete [] _inp_streams;
	delete [] _out_streams;
	delete [] _device_id;
	Free_Host_Memory();
	if (_pipes != 0L)
	{
		for (int i = 0;  i < _num_pipes;  ++i)
		{
			if (_pipes[i] != 0L) delete _pipes[i];
			_pipes[i] = 0L;
		}
		delete [] _pipes;
		_pipes = 0L;
	}
}

void Elastic_Propagator::Read_Earth_Model()
{
	_job->_Read_Earth_Model(this);
}

void Elastic_Propagator::Set_EM_Cell(
	int x,
	int y,
	int z,
	unsigned int word0,
	unsigned int word1,
	unsigned int word2,
	unsigned int word3
	)
{
	int xblk = x >> 2;
	int xidx = x & 3;

	if (xblk < 0 || xblk >= _NbX || y < 0 || y >= _ny || z < 0 || z >= _nz)
	{
		printf("Elastic_Propagator::Set_EM_Cell - Out of bounds - x=%d,y=%d,z=%d\n",x,y,z);
		exit(0);
	}
	
	int one_wf_size_f = 4 * _nz;
        int one_y_size_f = one_wf_size_f * 4;

	int idx = one_y_size_f * (size_t)y + (size_t)z * 4 + xidx;

	//if (y==720) printf("x-y-z=%d-%d-%d :: xblk=%ld, xidx=%ld, one_wf_size_f=%ld, one_y_size_f=%ld, idx=%ld, words=%d,%d,%d,%d\n",x,y,z,xblk,xidx,one_wf_size_f,one_y_size_f,idx,word0,word1,word2,word3);
	
	((unsigned int*)_EM[xblk])[idx                ] = word0;
	((unsigned int*)_EM[xblk])[idx+  one_wf_size_f] = word1;
	((unsigned int*)_EM[xblk])[idx+2*one_wf_size_f] = word2;
	((unsigned int*)_EM[xblk])[idx+3*one_wf_size_f] = word3;
}

void Elastic_Propagator::Get_EM_Cell(
	int x,
	int y,
	int z,
	unsigned int& word0,
        unsigned int& word1,
        unsigned int& word2,
        unsigned int& word3
        )
{
        int xblk = x >> 2;
        int xidx = x & 3;

	if (xblk < 0 || xblk >= _NbX || y < 0 || y >= _ny || z < 0 || z >= _nz)
	{
		printf("Elastic_Propagator::Get_EM_Cell - Out of bounds - x=%d,y=%d,z=%d\n",x,y,z);
		exit(0);
	}

        int one_wf_size_f = 4 * _nz;
        int one_y_size_f = one_wf_size_f * 4;

        int idx = one_y_size_f * (int)y + (int)z * 4 + xidx;

        word0 = ((unsigned int*)_EM[xblk])[idx                ];
        word1 = ((unsigned int*)_EM[xblk])[idx+  one_wf_size_f];
        word2 = ((unsigned int*)_EM[xblk])[idx+2*one_wf_size_f];
        word3 = ((unsigned int*)_EM[xblk])[idx+3*one_wf_size_f];
	//printf("x-y-z=%d,%d,%d :: words=%d,%d,%d,%d\n",x,y,z,word0,word1,word2,word3);
}

//
// Extract a receiver value from the wavefields on the host side.
// This is for debug purposes.
//
float Elastic_Propagator::Get_Receiver_Value(int wf_type, int x, int y, int z)
{
	int xblk = x >> 2;
        int xidx = x & 3;

        if (xblk < 0 || xblk >= _NbX || y < 0 || y >= _ny || z < 0 || z >= _nz)
        {
                printf("Elastic_Propagator::Get_Receiver_Value - Out of bounds - x=%d,y=%d,z=%d\n",x,y,z);
                exit(0);
        }

        int one_wf_size_f = 4 * _nz;
        int one_y_size_f = one_wf_size_f * 6;

        int idx = one_y_size_f * (int)y + (int)z * 4 + xidx;

	switch (wf_type)
	{
	case 0: // Vx
		return ((float*)_PV[xblk])[idx];
	case 1: // Vy
		return ((float*)_PV[xblk])[idx+one_wf_size_f];
	case 2: // Vz
		return ((float*)_PV[xblk])[idx+2*one_wf_size_f];
	case 3: // P
		{
			float txx = ((float*)_ST[xblk])[idx];
			float tyy = ((float*)_ST[xblk])[idx+one_wf_size_f];
			float tzz = ((float*)_ST[xblk])[idx+2*one_wf_size_f];
			return -(txx+tyy+tzz)/3.0f;
		}
	case 6: // txx
		return ((float*)_ST[xblk])[idx];
	case 7: // tyy
		return ((float*)_ST[xblk])[idx+one_wf_size_f];
	case 8: // tzz
		return ((float*)_ST[xblk])[idx+2*one_wf_size_f];
	case 9: // txy
		return ((float*)_ST[xblk])[idx+3*one_wf_size_f];
	case 10: // txz
		return ((float*)_ST[xblk])[idx+4*one_wf_size_f];
	case 11: // tyz
		return ((float*)_ST[xblk])[idx+5*one_wf_size_f];
	default:
		return 0.0f;
	}
}

void Elastic_Propagator::Set_WF_Value(int wf_type, int x, int y, int z, float val)
{
	int xblk = x >> 2;
        int xidx = x & 3;

        if (xblk < 0 || xblk >= _NbX || y < 0 || y >= _ny || z < 0 || z >= _nz)
        {
                printf("Elastic_Propagator::Set_WF_Value - Out of bounds - x=%d,y=%d,z=%d\n",x,y,z);
                exit(0);
        }

        int one_wf_size_f = 4 * _nz;
        int one_y_size_f = one_wf_size_f * 6;

        int idx = one_y_size_f * (int)y + (int)z * 4 + xidx;

        switch (wf_type)
        {
        case 0: // Vx
                ((float*)_PV[xblk])[idx] = val;
		break;
        case 1: // Vy
                ((float*)_PV[xblk])[idx+one_wf_size_f] = val;
		break;
        case 2: // Vz
                ((float*)_PV[xblk])[idx+2*one_wf_size_f] = val;
		break;
        case 3: // P
		((float*)_ST[xblk])[idx] = -val;
		((float*)_ST[xblk])[idx+one_wf_size_f] = -val;
		((float*)_ST[xblk])[idx+2*one_wf_size_f] = -val;
		break;
	case 6: // txx
		((float*)_ST[xblk])[idx] = val;
		break;
	case 7: // tyy
		((float*)_ST[xblk])[idx+one_wf_size_f] = val;
		break;
	case 8: // tzz
		((float*)_ST[xblk])[idx+2*one_wf_size_f] = val;
		break;
	case 9: // txy
		((float*)_ST[xblk])[idx+3*one_wf_size_f] = val;
		break;
	case 10: // txz
		((float*)_ST[xblk])[idx+4*one_wf_size_f] = val;
		break;
	case 11: // tyz
		((float*)_ST[xblk])[idx+5*one_wf_size_f] = val;
		break;
	default:
		break;
        }
}

//
// This method is called by Elastic_Modeling_Job object.
//
void Elastic_Propagator::_Insert_Earth_Model_Stripe(
	unsigned int* word0,
	unsigned int* word1,
	unsigned int* word2,
	unsigned int* word3,
	int n,
	int x0,
	int inc_x,
	int y0,
	int inc_y,
	int z0,
	int inc_z
	)
{
	for (int i = 0;  i < n;  ++i)
	{
		int x = x0 + i * inc_x;
		int y = y0 + i * inc_y;
		int z = z0 + i * inc_z;
		
		Set_EM_Cell(x,y,z,word0[i],word1[i],word2[i],word3[i]);
	}
}

unsigned int Elastic_Propagator::_Get_Earth_Model_Word(int widx, int x,int y,int z)
{
	unsigned int word[4];
	Get_EM_Cell(x,y,z,word[0],word[1],word[2],word[3]);
	return word[widx];
}

void Elastic_Propagator::_NABC_TOP_Extend(int z0)
{
	bool do_lo = z0 > 0 ? true : false;
	if (do_lo)
	{
		if (_log_level > 3) printf("NABC_TOP_Extend %d => [%d,%d]\n",z0,z0-1,0);
#pragma omp parallel for schedule(static,4)
		for (int x = 0;  x < _nx;  ++x)
		{
			for (int y = 0;  y < _ny;  ++y)
			{
				unsigned int word0, word1, word2, word3;
				Get_EM_Cell(x,y,z0,word0,word1,word2,word3);
				for (int z = 0;  z < z0;  ++z)
				{
					Set_EM_Cell(x,y,z,word0,word1,word2,word3);
				}
			}
		}
	}
}

void Elastic_Propagator::_NABC_BOT_Extend(int z1)
{
	bool do_hi = z1 < _nz-1 ? true : false;
	if (do_hi)
	{
		if (_log_level > 3) printf("NABC_BOT_Extend %d => [%d,%d]\n",z1,z1+1,_nz-1);
#pragma omp parallel for schedule(static,4)
		for (int x = 0;  x < _nx;  ++x)
		{
			for (int y = 0;  y < _ny;  ++y)
			{
				unsigned int word0, word1, word2, word3;
				Get_EM_Cell(x,y,z1,word0,word1,word2,word3);
				for (int z = z1+1;  z < _nz;  ++z)
				{
					Set_EM_Cell(x,y,z,word0,word1,word2,word3);
				}
			}
		}
	}
}

void Elastic_Propagator::_NABC_SDX_Extend(int x0, int x1)
{
	bool do_lo = x0 > 0 ? true : false;
	bool do_hi = x1 < _nx-1 ? true : false;
	if (_log_level > 3)
	{
		if (do_lo) printf("NABC_SDX_Extend %d => [%d,%d]\n",x0,x0-1,0);
		if (do_hi) printf("NABC_SDX_Extend %d => [%d,%d]\n",x1,x1+1,_nx-1);
	}
	if (do_lo || do_hi)
	{
#pragma omp parallel for schedule(static,4)
		for (int y = 0;  y < _ny;  ++y)
		{
			for (int z = 0;  z < _nz;  ++z)
			{
				unsigned int word0, word1, word2, word3;
				if (do_lo)
				{
					Get_EM_Cell(x0,y,z,word0,word1,word2,word3);
					for (int x = 0;  x < x0;  ++x)
					{
						Set_EM_Cell(x,y,z,word0,word1,word2,word3);
					}
				}
				if (do_hi)
				{
					Get_EM_Cell(x1,y,z,word0,word1,word2,word3);
					for (int x = x1+1;  x < _nx;  ++x)
					{
						Set_EM_Cell(x,y,z,word0,word1,word2,word3);
					}
				}
			}
		}
	}
}

void Elastic_Propagator::_NABC_SDY_Extend(int y0, int y1)
{
	bool do_lo = y0 > 0 ? true : false;
	bool do_hi = y1 < _ny-1 ? true : false;
	if (_log_level > 3)
	{
		if (do_lo) printf("NABC_SDY_Extend %d => [%d,%d]\n",y0,y0-1,0);
		if (do_hi) printf("NABC_SDY_Extend %d => [%d,%d]\n",y1,y1+1,_ny-1);
	}
	if (do_lo || do_hi)
	{
#pragma omp parallel for schedule(static,4)
		for (int x = 0;  x < _nx;  ++x)
		{
			for (int z = 0;  z < _nz;  ++z)
			{
				unsigned int word0, word1, word2, word3;
				if (do_lo)
				{
					Get_EM_Cell(x,y0,z,word0,word1,word2,word3);
					for (int y = 0;  y < y0;  ++y)
					{
						Set_EM_Cell(x,y,z,word0,word1,word2,word3);
					}
				}
				if (do_hi)
				{
					for (int y = y1+1;  y < _ny;  ++y)
					{
						Set_EM_Cell(x,y,z,word0,word1,word2,word3);
					}
				}
			}
		}
	}
}

Elastic_Modeling_Job* Elastic_Propagator::Get_Job()
{
	return _job;
}

int Elastic_Propagator::Get_Stencil_Order()
{
	return _stencil_order;
}

int Elastic_Propagator::Get_Total_Number_Of_Timesteps()
{
	return _pipes[0]->Get_Total_Number_Of_Timesteps();
}

int Elastic_Propagator::Get_Device_Index(int device_id)
{
	for (int i = 0;  i < _num_devices;  ++i)
	{
		if (_device_id[i] == device_id)
		{
			return i;
		}
	}
	return -1;
}

void* Elastic_Propagator::Get_Block(int bX, bool Is_Model, bool Is_PV)
{
	if (bX >= 0 && bX < _NbX)
	{
		if (Is_Model)
		{
			return _EM[bX];
		}
		else if (Is_PV)
		{
			return _PV[bX];
		}
		else
		{
			return _ST[bX];
		}
	}
	return 0L;
}

void* Elastic_Propagator::Get_Host_Block(int bX, bool Is_Model, bool Is_PV, bool Is_Input)
{
	if (_pinned)
	{
		return Get_Block(bX,Is_Model,Is_PV);
	}
	else
	{
		if (Is_Input)
		{
			if (Is_Model)
			{
				return _pbuf_EM[0];
			}
			else
			{
				if (Is_PV)
				{
					return _pbuf_PV[0];
				}
				else
				{
					return _pbuf_ST[0];
				}
			}
		}
		else
		{
			if (Is_Model)
			{
				// model only has output buffer for debug sessions.
				return _debug ? _pbuf_EM_Out[1] : 0L;
			}
			else
			{
				if (Is_PV)
				{
					return _pbuf_PV_Out[1];
				}
				else
				{
					return _pbuf_ST_Out[1];
				}
			}
		}
	}
}

cudaStream_t Elastic_Propagator::Get_Compute_Stream(int device_id)
{
	int device_index = Get_Device_Index(device_id);
	if (device_index >= 0)
	{
		if (_cmp_streams[device_index] == 0L)
		{
			cudaSetDevice(device_id);
			cudaStreamCreate(&(_cmp_streams[device_index]));
		}
		return _cmp_streams[device_index];
	}
	printf("Warning! Get_Compute_Stream returned nil!\n");
	return 0L;
}

cudaStream_t Elastic_Propagator::Get_Input_Stream(int device_id)
{
	int device_index = Get_Device_Index(device_id);
	if (device_index >= 0)
	{
		if (_inp_streams[device_index] == 0L)
		{
			cudaSetDevice(device_id);
			cudaStreamCreate(&(_inp_streams[device_index]));
		}
		return _inp_streams[device_index];
	}
	printf("Warning! Get_Input_Stream returned nil!\n");
	return 0L;
}

cudaStream_t Elastic_Propagator::Get_Output_Stream(int device_id)
{
	int device_index = Get_Device_Index(device_id);
	if (device_index >= 0)
	{
		if (_out_streams[device_index] == 0L)
		{
			cudaSetDevice(device_id);
			cudaStreamCreate(&(_out_streams[device_index]));
		}
		return _out_streams[device_index];
	}
	printf("Warning! Get_Output_Stream returned nil!\n");
	return 0L;
}

void Elastic_Propagator::_Compare(char* src, char* dst, size_t len)
{
	for (size_t i = 0;  i < len;  ++i)
	{
		if (src[i] != dst[i])
		{
			printf("Error! - Blocks differ!\n");
			exit(0);
		}
	}
}

void Elastic_Propagator::_Find_Non_Zeros(char* dst, size_t len)
{
	size_t len_f = len / 4;
	for (size_t i = 0;  i < len_f;  ++i)
	{
		if (((float*)dst)[i] != 0.0f)
		{
			printf("val at idx=%ld is %f\n",i,((float*)dst)[i]);
		}
	}
}

void Elastic_Propagator::Copy_To_Pinned_Buffer(int input_block_offset, int output_block_offset)
{
	// only do this if main volume is not pinned
	if (!_pinned)
	{
		if (input_block_offset >= 0)
		{
			void* src_PV = Get_Block(input_block_offset, false, true);
			omp_memcpy(_pbuf_PV[1], src_PV, _blkSize_PV);
			//_Compare((char*)_pbuf_PV[1], (char*)src_PV, _blkSize_PV);
			//printf("input_block_offset=%d, src_PV=%p\n",input_block_offset,src_PV);

			void* src_ST = Get_Block(input_block_offset, false, false);
			omp_memcpy(_pbuf_ST[1], src_ST, _blkSize_ST);
			//_Compare((char*)_pbuf_ST[1], (char*)src_ST, _blkSize_ST);
			//_Find_Non_Zeros((char*)src_ST, _blkSize_ST);
			//_Find_Non_Zeros((char*)_pbuf_ST[1], _blkSize_ST);
			//printf("input_block_offset=%d, src_ST=%p\n",input_block_offset,src_ST);

			void* src_EM = Get_Block(input_block_offset, true, false);
			omp_memcpy(_pbuf_EM[1], src_EM, _blkSize_EM);
			//_Compare((char*)_pbuf_EM[1], (char*)src_EM, _blkSize_EM);
			//printf("input_block_offset=%d, src_EM=%p\n",input_block_offset,src_EM);
		}

		if (output_block_offset >= 0)
		{
			void* dst_PV_Out = Get_Block(output_block_offset, false, true);
			omp_memcpy(dst_PV_Out, _pbuf_PV_Out[0], _blkSize_PV);

			void* dst_ST_Out = Get_Block(output_block_offset, false, false);
			omp_memcpy(dst_ST_Out, _pbuf_ST_Out[0], _blkSize_ST);

			if (_debug)
			{
				void* dst_EM_Out = Get_Block(output_block_offset, true, false);
				omp_memcpy(dst_EM_Out, _pbuf_EM_Out[0], _blkSize_EM);
			}
		}
		
	//	printf("Elastic_Propagator::Copy_To_Pinned_Buffer - input_block_offset=%d, output_block_offset=%d\n",input_block_offset,output_block_offset);
	}
}

void Elastic_Propagator::Shift_Pinned_Buffer()
{
	if (!_pinned)
	{
		void* tmp = _pbuf_PV[0];
		_pbuf_PV[0] = _pbuf_PV[1];
		_pbuf_PV[1] = tmp;

		tmp = _pbuf_ST[0];
		_pbuf_ST[0] = _pbuf_ST[1];
		_pbuf_ST[1] = tmp;

		tmp = _pbuf_EM[0];
		_pbuf_EM[0] = _pbuf_EM[1];
		_pbuf_EM[1] = tmp;

		tmp = _pbuf_PV_Out[0];
		_pbuf_PV_Out[0] = _pbuf_PV_Out[1];
		_pbuf_PV_Out[1] = tmp;

		tmp = _pbuf_ST_Out[0];
		_pbuf_ST_Out[0] = _pbuf_ST_Out[1];
		_pbuf_ST_Out[1] = tmp;

		if (_debug)
		{
			tmp = _pbuf_EM_Out[0];
			_pbuf_EM_Out[0] = _pbuf_EM_Out[1];
			_pbuf_EM_Out[1] = tmp;
		}
	}
}

void Elastic_Propagator::Prepare_For_Propagation(Elastic_Shot* shot)
{
	const double courant_safe = 0.95;
 
	// determine internal timestepping
	// ..Courant# for O(2) time leap frog SG FD
	double courant = 1.0 / (sqrt(3.0) * (Elastic_Buffer::_C0 - Elastic_Buffer::_C1 + Elastic_Buffer::_C2 - Elastic_Buffer::_C3));
	courant = courant_safe * _job->Get_Courant_Factor() * courant;  // drop Courant# by this safety factor
	printf("Courant# = %f\n",courant);
	if (shot->Get_Ordertime() == 4) 
	{
		courant = courant*0.8/0.54;  // O(4) time Courant# for Omelyan symplectic
	}

	double dl_min = _job->Get_DX();
	if (_job->Get_DY() < dl_min) dl_min = _job->Get_DY();
	if (_job->Get_DZ() < dl_min) dl_min = _job->Get_DZ();
	if (_log_level > 3) printf("Smallest cell size is %lf\n",dl_min);

	if (_log_level > 3) printf("Determining max(Vp)...\n");
	float glob_max_Vp = 0.0f;
	float glob_min_Q = 1e36f, glob_max_Q = -1e36f;
#pragma omp parallel for
	for (int iz = 0;  iz < _job->Get_Propagation_NZ();  ++iz)
	{
		float my_max_Vp = 0.0f;
		float my_min_Q=1e36f, my_max_Q=-1e36f;
		for (int iy = 0;  iy < _job->Get_Propagation_NY();  ++iy)
		{
			for (int ix = 0;  ix < _job->Get_Propagation_NX();  ++ix)
			{
				// find max of Vp * Vp * ( 1 + 2 * Epsilon2 )
				float Vp = _job->Get_Earth_Model_Attribute(_job->Attr_Idx_Vp, ix, iy, iz);
				float Epsilon2 = _job->Get_Earth_Model_Attribute(_job->Attr_Idx_Epsilon2, ix, iy, iz);
				float C11Max = sqrt(Vp * Vp * ( 1.0f + 2.0f * Epsilon2 ));
				if (C11Max > my_max_Vp) my_max_Vp = C11Max;

				// find max of Vp * Vp
				float C33Max = Vp;
				if (C33Max > my_max_Vp) my_max_Vp = C33Max;

				// find max of Vp * Vp * ( 1 + 2 * Epsilon1 )
				float Epsilon1 = _job->Get_Earth_Model_Attribute(_job->Attr_Idx_Epsilon1, ix, iy, iz);
				float C22Max = sqrt(Vp * Vp * ( 1.0f + 2.0f * Epsilon1 ));
				if (C22Max > my_max_Vp) my_max_Vp = C22Max;

				float Q = _job->Get_Earth_Model_Attribute(_job->Attr_Idx_Q, ix, iy, iz);
				if (Q < my_min_Q) my_min_Q = Q;
				if (Q > my_max_Q) my_max_Q = Q;
			}
		}
#pragma omp criticial
		{
			if (my_max_Vp > glob_max_Vp) glob_max_Vp = my_max_Vp;
			if (my_min_Q < glob_min_Q) glob_min_Q = my_min_Q;
			if (my_max_Q > glob_max_Q) glob_max_Q = my_max_Q;
		}
	}
	printf("Vp max = %lf\n",glob_max_Vp);
	printf("Q=[%e,%e]\n",glob_min_Q,glob_max_Q);

	double dti_max = courant * dl_min / glob_max_Vp;
	_dti = dti_max;  // use time step determined by FD stability
	if (_log_level > 2) printf("Internal dt is %lfms\n",_dti*1e3);
	_num_timesteps = (int)ceil(shot->Get_Propagation_Time()/_dti);
	printf("%d timesteps.\n",_num_timesteps);

	shot->Prepare_Source_Wavelet(_dti);
	shot->Create_Receiver_Transfer_Buffers(this);
}

void Elastic_Propagator::Propagate_Shot(Elastic_Shot* shot)
{
	Prepare_For_Propagation(shot);
	while (!Propagate_One_Block(_num_timesteps, shot));
}

// returns true if at least Number_Of_Timesteps timesteps have been completed
bool Elastic_Propagator::Propagate_One_Block(int Number_Of_Timesteps, Elastic_Shot* shot)
{
	if (_pbuf_first_call)
	{
		Copy_To_Pinned_Buffer(_pipes[0]->Get_Input_Block_Offset(0),-1);
		clock_gettime(CLOCK_REALTIME, &_before);
		_pbuf_first_call = false;
	}
	else
	{
		for (int i = 0;  i < _num_pipes;  ++i) _pipes[i]->Shift_Buffers();
	}
	Shift_Pinned_Buffer();
	shot->Shift_Receiver_Transfer_Buffers();

	// launch data flow and compute kernels
	for (int i = 0;  i < _num_pipes;  ++i) _pipes[i]->Launch_Data_Transfers();
	//for (int i = 0;  i < _num_pipes;  ++i) _pipes[i]->Launch_Simple_Copy_Kernel();
	for (int i = 0;  i < _num_pipes;  ++i) _pipes[i]->Launch_Compute_Kernel(_dti,shot);

	Copy_To_Pinned_Buffer(_pipes[0]->Get_Input_Block_Offset(1),_pipes[0]->Get_Output_Block_Offset(-1));

	// wait for cuda streams
	for (int i = 0;  i < _num_devices;  ++i) if (_cmp_streams[i] != 0L) cudaStreamSynchronize(_cmp_streams[i]);
	for (int i = 0;  i < _num_devices;  ++i) if (_inp_streams[i] != 0L) cudaStreamSynchronize(_inp_streams[i]);
	for (int i = 0;  i < _num_devices;  ++i) if (_out_streams[i] != 0L) cudaStreamSynchronize(_out_streams[i]);

        gpuErrchk( cudaPeekAtLastError() );

	if ((_pinned && _pipes[0]->Get_Output_Block_Offset(1) == 0) || (!_pinned && _pipes[0]->Get_Output_Block_Offset(0) == 0))
	{
		// finished one iteration
		struct timespec after;
		clock_gettime(CLOCK_REALTIME, &after);
		double elapsed_time = (double)after.tv_sec + (double)after.tv_nsec * 1e-9 - (double)_before.tv_sec - (double)_before.tv_nsec * 1e-9;
		double mcells_per_s = (double)_nx * (double)_ny * (double)_nz * (double)Get_Total_Number_Of_Timesteps() * 1e-6 / elapsed_time;
		int ts = _pipes[0]->Get_Output_Block_Timestep(0) - Get_Total_Number_Of_Timesteps();
		if (ts == 0)
		{
			printf("LEAD-IN (filling up pipeline) took %.2f secs\n",elapsed_time);
		}
		else
		{
			printf("Timesteps %4d to %4d :: %.2f secs - %.0f MC/s\n",ts-Get_Total_Number_Of_Timesteps()+1,ts,elapsed_time,mcells_per_s);

			int iy = 411;
			int iz = 960;
			
			char path[4096];
			sprintf(path, "/panfs07/esdrd/tjhc/ELA_on_GPU/slices/xz_slice_Y=%04d_%04d_P",iy,ts);
			_job->Write_XZ_Slice(path, 3, iy);
			for (int iZZ = 0;  iZZ < 10;  ++iZZ)
			{
				int loc_iz = iz - iZZ * 50;
				sprintf(path, "/panfs07/esdrd/tjhc/ELA_on_GPU/slices/xy_slice_Z=%04d_%04d_P",loc_iz,ts);
				_job->Write_XY_Slice(path, 3, loc_iz);
			}

			/*
			sprintf(path, "/panfs07/esdrd/tjhc/ELA_on_GPU/slices/xz_slice_%04d_Vx", ts);
			_job->Write_XZ_Slice(path, 0, iy);
			sprintf(path, "/panfs07/esdrd/tjhc/ELA_on_GPU/slices/xz_slice_%04d_Vy", ts);
			_job->Write_XZ_Slice(path, 1, iy);
			sprintf(path, "/panfs07/esdrd/tjhc/ELA_on_GPU/slices/xz_slice_%04d_Vz", ts);
			_job->Write_XZ_Slice(path, 2, iy);
			sprintf(path, "/panfs07/esdrd/tjhc/ELA_on_GPU/slices/xz_slice_%04d_P", ts);
			_job->Write_XZ_Slice(path, 3, iy);

			sprintf(path, "/panfs07/esdrd/tjhc/ELA_on_GPU/slices/xz_slice_%04d_Txx", ts);
			_job->Write_XZ_Slice(path, 6, iy);
			sprintf(path, "/panfs07/esdrd/tjhc/ELA_on_GPU/slices/xz_slice_%04d_Tyy", ts);
			_job->Write_XZ_Slice(path, 7, iy);
			sprintf(path, "/panfs07/esdrd/tjhc/ELA_on_GPU/slices/xz_slice_%04d_Tzz", ts);
			_job->Write_XZ_Slice(path, 8, iy);
			sprintf(path, "/panfs07/esdrd/tjhc/ELA_on_GPU/slices/xz_slice_%04d_Txy", ts);
			_job->Write_XZ_Slice(path, 9, iy);
			sprintf(path, "/panfs07/esdrd/tjhc/ELA_on_GPU/slices/xz_slice_%04d_Txz", ts);
			_job->Write_XZ_Slice(path, 10, iy);
			sprintf(path, "/panfs07/esdrd/tjhc/ELA_on_GPU/slices/xz_slice_%04d_Tyz", ts);
			_job->Write_XZ_Slice(path, 11, iy);
			*/
		}
		clock_gettime(CLOCK_REALTIME, &_before);
		return ts >= Number_Of_Timesteps ? true : false;
	}
	return false;
}

void Elastic_Propagator::Free_Host_Memory()
{
	if (_ts != 0L)
	{
		delete [] _ts;
		_ts = 0L;
	}
	if (_PV != 0L)
	{
		for (int i = 0;  i < _NbX;  ++i)
		{
			if (_PV[i] != 0L)
			{
				if (_pinned)
					cudaFreeHost(_PV[i]);
				else
					free(_PV[i]);
				_PV[i] = 0L;
			}
		}
		delete [] _PV;
		_PV = 0L;
	}
	if (_ST != 0L)
	{
		for (int i = 0;  i < _NbX;  ++i)
		{
			if (_ST[i] != 0L)
			{
				if (_pinned)
					cudaFreeHost(_ST[i]);
				else
					free(_ST[i]);
				_ST[i] = 0L;
			}
		}
		delete [] _ST;
		_ST = 0L;
	}
	if (_EM != 0L)
        {
                for (int i = 0;  i < _NbX;  ++i)
                {
                        if (_EM[i] != 0L)
                        {
				if (_pinned)
                                	cudaFreeHost(_EM[i]);
				else
					free(_EM[i]);
                                _EM[i] = 0L;
                        }
                }
                delete [] _EM;
                _EM = 0L;
        }
	if (!_pinned)
	{
		// deallocate transfer buffers
		if (_pbuf_PV != 0L)
		{
			if (_pbuf_PV[0] != 0L) cuda_host_free(_pbuf_PV[0]);
			if (_pbuf_PV[1] != 0L) cuda_host_free(_pbuf_PV[1]);
			delete [] _pbuf_PV;
			_pbuf_PV = 0L;
		}
		if (_pbuf_ST != 0L)
		{
			if (_pbuf_ST[0] != 0L) cuda_host_free(_pbuf_ST[0]);
			if (_pbuf_ST[1] != 0L) cuda_host_free(_pbuf_ST[1]);
			delete [] _pbuf_ST;
			_pbuf_ST = 0L;
		}
		if (_pbuf_EM != 0L)
		{
			if (_pbuf_EM[0] != 0L) cuda_host_free(_pbuf_EM[0]);
			if (_pbuf_EM[1] != 0L) cuda_host_free(_pbuf_EM[1]);
			delete [] _pbuf_EM;
			_pbuf_EM = 0L;
		}
		if (_pbuf_PV_Out != 0L)
		{
			if (_pbuf_PV_Out[0] != 0L) cuda_host_free(_pbuf_PV_Out[0]);
			if (_pbuf_PV_Out[1] != 0L) cuda_host_free(_pbuf_PV_Out[1]);
			delete [] _pbuf_PV_Out;
			_pbuf_PV_Out = 0L;
		}
		if (_pbuf_ST_Out != 0L)
		{
			if (_pbuf_ST_Out[0] != 0L) cuda_host_free(_pbuf_ST_Out[0]);
			if (_pbuf_ST_Out[1] != 0L) cuda_host_free(_pbuf_ST_Out[1]);
			delete [] _pbuf_ST_Out;
			_pbuf_ST_Out = 0L;
		}
		if (_debug && _pbuf_EM_Out != 0L)
		{
			if (_pbuf_EM_Out[0] != 0L) cuda_host_free(_pbuf_EM_Out[0]);
			if (_pbuf_EM_Out[1] != 0L) cuda_host_free(_pbuf_EM_Out[1]);
			delete [] _pbuf_EM_Out;
			_pbuf_EM_Out = 0L;
		}
	}
}

#define NUM_PAGES 1

void Elastic_Propagator::omp_memclear(void* dst, size_t len)
{
        size_t leni = len / 16;
        size_t nn = (len + NUM_PAGES*getpagesize()-1) / (NUM_PAGES*getpagesize());
	size_t One_Thread_Full = NUM_PAGES * getpagesize() / sizeof(__m128);
	//printf("omp_memclear len=%ld, leni=%ld, nn=%ld, One_Thread_Full=%ld\n",len,leni,nn,One_Thread_Full);
#pragma omp parallel for schedule(static)
        for (int i = 0;  i < nn;  ++i)
        {
                __m128 zero = _mm_set_ps(0.0f, 0.0f, 0.0f, 0.0f);
                size_t i0 = (size_t)i * One_Thread_Full;
                size_t in = leni - i0;
                if (in > One_Thread_Full) in = One_Thread_Full;
                __m128* d = (__m128*)dst + i0;
                for (int j = 0;  j < in;  ++j)
                {
                        _mm_stream_ps((float*)(d+j), zero);
                }
        }
}

void Elastic_Propagator::omp_memcpy(void* dst, void* src, size_t len)
{
        size_t leni = len / 16;
        size_t nn = (len + NUM_PAGES*getpagesize()-1) / (NUM_PAGES*getpagesize());
	size_t One_Thread_Full = NUM_PAGES * getpagesize() / sizeof(__m128);
#pragma omp parallel for schedule(static)
        for (int i = 0;  i < nn;  ++i)
        {
                size_t i0 = (size_t)i * One_Thread_Full;
                size_t in = leni - i0;
                if (in > One_Thread_Full) in = One_Thread_Full;
                __m128* d = (__m128*)dst + i0;
                __m128* s = (__m128*)src + i0;
                for (int j = 0;  j < in;  ++j)
                {
                        _mm_stream_ps((float*)(d+j),_mm_load_ps((float*)(s+j)));
                }
        }
}

// allocate memory,
// initialize with omp_memclear so that all pages end up in local memory,
// pin pages and register with CUDA
void Elastic_Propagator::cuda_host_memalign(void** p, size_t alignment, size_t len)
{
        posix_memalign((void**)p, alignment, len);
        omp_memclear((void*)(*p), len);
        cudaHostRegister((void*)(*p), len, 0);
}

// unregister with CUDA,
// deallocate
void Elastic_Propagator::cuda_host_free(void* p)
{
        cudaHostUnregister(p);
        free(p);
}

void Elastic_Propagator::Allocate_Host_Memory(bool Pinned, bool Patterned)
{
	Free_Host_Memory();
	_pinned = Pinned;
	_ts = new int[_NbX];
	_PV = new void*[_NbX];
	_ST = new void*[_NbX];
	_EM = new void*[_NbX];
	_blkSize = (size_t)_bsX * (size_t)_ny * (size_t)_nz;
	_blkSize_PV = _blkSize * (size_t)24;
	_blkSize_ST = _blkSize * (size_t)24;
	_blkSize_EM = _blkSize * (size_t)16;
	_blkSize_PV = ((_blkSize_PV + getpagesize() - 1) / getpagesize()) * getpagesize();
	_blkSize_ST = ((_blkSize_ST + getpagesize() - 1) / getpagesize()) * getpagesize();
	_blkSize_EM = ((_blkSize_EM + getpagesize() - 1) / getpagesize()) * getpagesize();
	printf("_NbX=%d, _bsX=%d, _ny=%d, _nz=%d\n",_NbX,_bsX,_ny,_nz);
	printf("_blkSize_PV=%ld, _blkSize_ST=%ld, _blkSize_EM=%ld\n",_blkSize_PV,_blkSize_ST,_blkSize_EM);
	printf("Allocating %s memory...\n",Pinned?"PINNED":"REGULAR");
	for (int i = 0;  i < _NbX;  ++i)
	{
		if (Pinned)
		{
			cudaError_t err1 = cudaHostAlloc(&(_PV[i]),_blkSize_PV,cudaHostAllocDefault);
			if (err1 != cudaSuccess) _PV[i] = 0L;

			cudaError_t err2 = cudaHostAlloc(&(_ST[i]),_blkSize_ST,cudaHostAllocDefault);
			if (err2 != cudaSuccess) _ST[i] = 0L;

			cudaError_t err3 = cudaHostAlloc(&(_EM[i]),_blkSize_EM,cudaHostAllocDefault);
			if (err3 != cudaSuccess) _EM[i] = 0L;
		}
		else
		{
			posix_memalign((void**)&(_PV[i]), getpagesize(), _blkSize_PV);
			omp_memclear(_PV[i], _blkSize_PV);

			posix_memalign((void**)&(_ST[i]), getpagesize(), _blkSize_ST);
			omp_memclear(_ST[i], _blkSize_ST);

			posix_memalign((void**)&(_EM[i]), getpagesize(), _blkSize_EM);
			omp_memclear(_EM[i], _blkSize_EM);
		}
	}
	if (!Pinned)
	{
		// allocate pinned transfer buffers
		_pbuf_PV = new void*[2];
		cuda_host_memalign(&(_pbuf_PV[0]), getpagesize(), _blkSize_PV);
		cuda_host_memalign(&(_pbuf_PV[1]), getpagesize(), _blkSize_PV);

		_pbuf_ST = new void*[2];
		cuda_host_memalign(&(_pbuf_ST[0]), getpagesize(), _blkSize_ST);
		cuda_host_memalign(&(_pbuf_ST[1]), getpagesize(), _blkSize_ST);

		_pbuf_EM = new void*[2];
		cuda_host_memalign(&(_pbuf_EM[0]), getpagesize(), _blkSize_EM);
		cuda_host_memalign(&(_pbuf_EM[1]), getpagesize(), _blkSize_EM);

		_pbuf_PV_Out = new void*[2];
		cuda_host_memalign(&(_pbuf_PV_Out[0]), getpagesize(), _blkSize_PV);
		cuda_host_memalign(&(_pbuf_PV_Out[1]), getpagesize(), _blkSize_PV);

		_pbuf_ST_Out = new void*[2];
		cuda_host_memalign(&(_pbuf_ST_Out[0]), getpagesize(), _blkSize_ST);
		cuda_host_memalign(&(_pbuf_ST_Out[1]), getpagesize(), _blkSize_ST);

		if (_debug)
		{
			_pbuf_EM_Out = new void*[2];
			cuda_host_memalign(&(_pbuf_EM_Out[0]), getpagesize(), _blkSize_EM);
			cuda_host_memalign(&(_pbuf_EM_Out[1]), getpagesize(), _blkSize_EM);
		}
	}
	if (_debug)
	{
		// initialize with pattern, used for debugging
		size_t blkSize_PV_l = _blkSize_PV / 8;
		for (size_t blk = 0;  blk < _NbX;  ++blk)
		{
			for (size_t idx = 0;  idx < blkSize_PV_l;  ++idx)
			{
				((long**)_PV)[blk][idx] = blk * blkSize_PV_l + idx;
			}
		}

		size_t blkSize_ST_l = _blkSize_ST / 8;
                for (size_t blk = 0;  blk < _NbX;  ++blk)
                {
                        for (size_t idx = 0;  idx < blkSize_ST_l;  ++idx)
                        {
                                ((long**)_ST)[blk][idx] = blk * blkSize_ST_l + idx;
                        }
                }

		size_t blkSize_EM_l = _blkSize_EM / 8;
                for (size_t blk = 0;  blk < _NbX;  ++blk)
                {
                        for (size_t idx = 0;  idx < blkSize_EM_l;  ++idx)
                        {
                                ((long**)_EM)[blk][idx] = blk * blkSize_EM_l + idx;
                        }
                }
	}
}

bool Elastic_Propagator::Check_Host_Memory()
{
	if (!_debug)
	{
		return true;
	}
	else
	{
		bool Error = false;

		size_t blkSize_PV_l = _blkSize_PV / 8;
		for (size_t blk = 0;  blk < _NbX && !Error;  ++blk)
		{
			for (size_t idx = 0;  idx < blkSize_PV_l && !Error;  ++idx)
			{
				size_t val = blk * blkSize_PV_l + idx;
				if (((long**)_PV)[blk][idx] != val)
				{
					Error = true;
					printf("Error (PV)! Expected %ld, found %ld at blk:idx=%ld:%ld\n",val,((long**)_PV)[blk][idx],blk,idx);
				}
			}
		}
		if (!Error)
		{
			printf("PV host memory test PASSED!\n");

			size_t blkSize_ST_l = _blkSize_ST / 8;
			for (size_t blk = 0;  blk < _NbX && !Error;  ++blk)
			{
				for (size_t idx = 0;  idx < blkSize_ST_l && !Error;  ++idx)
				{
					size_t val = blk * blkSize_ST_l + idx;
					if (((long**)_ST)[blk][idx] != val)
					{
						Error = true;
						printf("Error (ST)! Expected %ld, found %ld at blk:idx=%ld:%ld\n",val,((long**)_ST)[blk][idx],blk,idx);
					}
				}
			}
			if (!Error)
			{
				printf("ST host memory test PASSED!\n");

				size_t blkSize_EM_l = _blkSize_EM / 8;
				for (size_t blk = 0;  blk < _NbX && !Error;  ++blk)
				{
					for (size_t idx = 0;  idx < blkSize_EM_l && !Error;  ++idx)
					{
						size_t val = blk * blkSize_EM_l + idx;
						if (((long**)_EM)[blk][idx] != val)
						{
							Error = true;
							printf("Error (EM)! Expected %ld, found %ld at blk:idx=%ld:%ld\n",val,((long**)_EM)[blk][idx],blk,idx);
						}
					}
				}
				if (!Error)
				{
					printf("EM host memory test PASSED!\n");
				}
			}
		}

		return !Error;
	}
}

void Elastic_Propagator::Free_Device_Memory()
{
	for (int i = 0;  i < _num_pipes;  ++i)
	{
		_pipes[i]->Free_Device_Memory();
	}
}

void Elastic_Propagator::Allocate_Device_Memory()
{
	// no need to explicitly free device memory, each call to this function on the buffer objects frees its memory.
	for (int i = 0;  i < _num_pipes;  ++i)
        {
                _pipes[i]->Allocate_Device_Memory();
        }
}

int Elastic_Propagator::Get_Block_Size_X()
{
	return _bsX;
}

int Elastic_Propagator::Get_Number_Of_Blocks()
{
	return _NbX;
}

bool Elastic_Propagator::Enable_Peer_Access(int device_id, int peer_device_id)
{
	int device_index = Get_Device_Index(device_id);
	int peer_device_index = Get_Device_Index(peer_device_id);
	if (!_tried_p2p[device_index][peer_device_index])
	{
		_tried_p2p[device_index][peer_device_index] = true;
		
		int yes_sir;
		/*
		gpuErrchk( cudaDeviceCanAccessPeer(&yes_sir,device_id,peer_device_id) );
		if (yes_sir)
		{
			cudaSetDevice(device_id);
			gpuErrchk( cudaDeviceEnablePeerAccess(peer_device_id,0) );
			printf("Enabled peer access for device %d to device %d\n",device_id,peer_device_id);
		}
		*/
	
		gpuErrchk( cudaDeviceCanAccessPeer(&yes_sir,peer_device_id,device_id) );
		if (yes_sir)
		{
			cudaSetDevice(peer_device_id);
			gpuErrchk( cudaDeviceEnablePeerAccess(device_id,0) );
			printf("Enabled peer access for device %d to device %d\n",peer_device_id,device_id);
		}
	}
}

int Elastic_Propagator::Get_Number_Of_Pipelines()
{
	return _num_pipes;
}

Elastic_Pipeline* Elastic_Propagator::Get_Pipeline(int pipe_idx)
{
	if (pipe_idx >= 0 && pipe_idx < _num_pipes)
	{
		return _pipes[pipe_idx];
	}
	return 0L;
}

void Elastic_Propagator::Print_Graphical()
{
	for (int i = 0;  i < _num_pipes;  ++i)
	{
		_pipes[i]->Print_Graphical();
	}
}

double Elastic_Propagator::Calculate_Cost(int y0, int ylen, int ny, int num_timesteps, int GPUs_per_pipe, int half_stencil, double* rel_cost)
{
	int yy0 = y0;
	int yy1 = y0 + ylen - 1;
	double cost = 0.0;
	for (int iGPU = GPUs_per_pipe-1;  iGPU >= 0;  --iGPU)
	{
		for (int iStep = num_timesteps-1;  iStep >= 0;  --iStep)
		{
			for (int i = 0;  i < 2;  ++i)
			{
				// substeps are reversed with regards to index 'i'
				cost += (double)(yy1 - yy0 + 1) * rel_cost[1-i];
				yy0 -= half_stencil;
				yy1 += half_stencil;
				if (yy0 < 0) yy0 = 0;
				if (yy1 >= ny) yy1 = ny - 1;
			}
		}
	}
	return cost;
}

bool Elastic_Propagator::Print_Device_Stats(int device_id, double& TFLOPS, double& GB_per_s)
{
	TFLOPS = GB_per_s = 0.0;
	cudaDeviceProp devProps;
	cudaError_t err = cudaGetDeviceProperties(&devProps, device_id);
	if (err == cudaSuccess)
	{
		cudaSetDevice(device_id);
		size_t free,total;
		err = cudaMemGetInfo(&free,&total);
		if (err == cudaSuccess)
		{
			double dFree_MB = (double)free / 1048576.0;
			GB_per_s = (double)devProps.memoryBusWidth * (double)devProps.memoryClockRate / 4e6;
			int Cores_per_SM;
			if (devProps.major == 1)
			{
				Cores_per_SM = 8;
			}
			else if (devProps.major == 2)
			{
				if (devProps.minor == 1)
				{
					Cores_per_SM = 48;
				}
				else
				{
					Cores_per_SM = 32;
				}
			}
			else if (devProps.major == 3)
			{
				Cores_per_SM = 192;
			}
			TFLOPS = (double)devProps.clockRate * (double)devProps.multiProcessorCount * (double)Cores_per_SM / 5e8;
			printf("device_id %d :: %s, CC=%d.%d, Free Mem=%.2f MB, %.3f TFLOPS, %.0f GB/s\n",device_id,devProps.name,devProps.major,devProps.minor,dFree_MB,TFLOPS,GB_per_s);
			return true;
		}
	}
	return false;
}

bool Elastic_Propagator::Check_GPUs(int* device_id, int num_devices)
{
	printf("\n");
	int device_count = 0;
	cudaGetDeviceCount(&device_count);
	if (device_count < 1)
	{
		printf("No CUDA capable devices found!\n\n");
		return false;
	}
	double Total_GB_per_s=0.0, Total_TFLOPS=0.0;
	for (int i = 0;  i < num_devices;  ++i)
	{
		double GB_per_s, TFLOPS;
		if (!Print_Device_Stats(device_id[i],TFLOPS,GB_per_s))
		{
			printf("device_id %d not found\n\n",device_id[i]);
			return false;
		}
		Total_GB_per_s += GB_per_s;
		Total_TFLOPS += TFLOPS;
	}
	printf("Aggregate %.3f TFLOPS, %.0f GB/s\n\n",Total_TFLOPS,Total_GB_per_s);
	return true;
}

double Elastic_Propagator::Get_Relative_Cost(bool Is_PV)
{
	if (Is_PV)
		return _rel_cost[1];
	else
		return _rel_cost[0];
}

double Elastic_Propagator::Get_Minimum_Workload()
{
	double ylen = (double)_ny;
	return (double)Get_Total_Number_Of_Timesteps() * ( Get_Relative_Cost(false) * ylen + Get_Relative_Cost(true) * ylen ) / (double)_num_pipes;
}

void Elastic_Propagator::Add_H2D(unsigned long len)
{
	_h2d += len;
}

void Elastic_Propagator::Add_D2H(unsigned long len)
{
	_d2h += len;
}

bool Elastic_Propagator::Is_Debug()
{
	return _debug;
}
