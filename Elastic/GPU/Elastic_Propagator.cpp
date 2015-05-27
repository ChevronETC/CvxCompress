#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <xmmintrin.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>

#include "gpuAssert.h"
#include "Elastic_Buffer.hxx"
#include "Elastic_Pipeline.hxx"
#include "Elastic_Propagator.hxx"
#include "Elastic_Modeling_Job.hxx"
#include "Elastic_Shot.hxx"
#include "Elastic_SEGY_File.hxx"
#include "Global_Coordinate_System.hxx"
#include "Voxet.hxx"

//Un-comment this to do more detailed timing of the various sections
//of the propagate one block routine
#define DETAILED_TIMING

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
			false
	     );
}

// Determine number of physical CPU cores.
// Hyper-threaded logical cores are not counted.
// Cache_Size is per core and in whole KB.
int Elastic_Propagator::Get_Physical_Core_Count(int& Cache_Size_Per_Core_KB)
{
        FILE* fp = fopen("/proc/cpuinfo","r");
        if (fp != 0L)
        {
                char buf[256];
                int max_phys_id = -1, max_cores = -1, max_cache_size = -1;
                while (fgets(buf, 256, fp) != 0L)
                {
                        int phys_id = -1;
                        if (sscanf(buf, "physical id : %d", &phys_id) == 1)
                        {
                                if (phys_id > max_phys_id) max_phys_id = phys_id;
                        }

                        int cpu_cores = -1;
                        if (sscanf(buf, "cpu cores : %d", &cpu_cores) == 1)
                        {
                                if (cpu_cores > max_cores) max_cores = cpu_cores;
                        }

                        int cache_size = -1;
                        if (sscanf(buf, "cache size : %d", &cache_size) == 1)
                        {
                                if (cache_size > max_cache_size) max_cache_size = cache_size;
                        }
                }
                fclose(fp);
                if (max_phys_id >= 0 && max_cores > 0)
                {
                        if (max_cache_size >= 0)
                        {
                                Cache_Size_Per_Core_KB = max_cache_size / max_cores;
                        }
                        else
                        {
                                Cache_Size_Per_Core_KB = -1;
                        }
                        return (max_phys_id+1) * max_cores;
                }
                else
                {
                        return -1;
                }
        }
        return -1;
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
	bool debug
	)
{
	int Cache_Size_Per_Core_KB;
	int num_cores = Get_Physical_Core_Count(Cache_Size_Per_Core_KB);
	if (num_cores > 0)
	{
		int num_threads = num_cores - 2; // num_cores >= 16 ? 12 : num_cores - 4;
		printf("Elastic_Propagator::_init - Machine has %d physical cores. Using %d threads for openmp.\n",num_cores,num_threads);
		omp_set_num_threads(num_threads);
	}

	_log_level = log_level;
	_best_num_z = -1;
	_num_z = 0L;
	_num_z_throughput = 0L;
	_num_num_z = 0;
	_curr_num_z = 0;
	_dti = 0.0;
	_slow_data_transfers = false;
	_timer1 = 0.0;
	_timer2 = 0.0;
	_timer3 = 0.0;
	_timer4 = 0.0;
	_timer5 = 0.0;
	_h2d = 0;
	_d2h = 0;
	_h2h = 0;
	_prev_h2d = 0;
	_prev_d2h = 0;
	_prev_h2h = 0;
	_num_timesteps = 0;
	_job = job;
	_debug = debug;
	_num_devices = 0;
	_device_id = 0L;
	_num_pipes = 0;
	_tried_p2p = 0L;
	_cmp_streams = 0L;
	_inp_streams = 0L;
	_out_streams = 0L;
	_rxx_streams = 0L;

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
	_PV_pinned = 0L;
	_ST_pinned = 0L;
	_EM_pinned = 0L;
	_num_pinned_PV = 0;
	_num_pinned_ST = 0;
	_num_pinned_EM = 0;
	_pbuf_PV = 0L;
	_pbuf_ST = 0L;
	_pbuf_EM = 0L;
	_pbuf_PV_Out = 0L;
	_pbuf_ST_Out = 0L;
	_pbuf_EM_Out = 0L;

	_pbuf_first_call = true;

	// ..determine relative cost of two kernels. For now, this is just hard-coded
	_rel_cost = new double[2];
	_rel_cost[0] = 0.5;  // T kernel
	_rel_cost[1] = 0.5;  // V kernel
}

Elastic_Propagator::~Elastic_Propagator()
{
	Free_Host_Memory();
	Delete_Compute_Pipelines();
	Free_Device_Memory();
}

bool Elastic_Propagator::Build_Compute_Pipelines(
	int num_pipes, 
	int num_timesteps, 
	const int* device_id, 
	int num_devices,
	bool partial_allowed
	)
{
	bool load_balance_possible = true;
	_num_devices = num_devices;
	_device_id = new int[_num_devices];
	_cmp_streams = new cudaStream_t[_num_devices];
	_inp_streams = new cudaStream_t[_num_devices];
	_out_streams = new cudaStream_t[_num_devices];
	_rxx_streams = new cudaStream_t[_num_devices];
	for (int i = 0;  i < _num_devices;  ++i)
	{
		_device_id[i] = device_id[i];
		_cmp_streams[i] = 0L;
		_inp_streams[i] = 0L;
		_out_streams[i] = 0L;
		_rxx_streams[i] = 0L;
	}

	_tried_p2p = new bool*[num_devices];
	for (int i = 0;  i < num_devices;  ++i)
	{
		_tried_p2p[i] = new bool[num_devices];
		for (int j = 0;  j < num_devices;  ++j)
		{
			_tried_p2p[i][j] = (i == j) ? true : false;
		}
	}

	if (Check_GPUs(_device_id, num_devices))
	{
		_num_pipes = num_pipes;
		_GPUs_per_pipe = num_devices / num_pipes;
		int half_stencil = _stencil_order / 2;
		int z0 = 0;
		int z1 = _nz-1;

		// load balancing
		int* pipe_width = new int[num_pipes];

		// ..between pipes
		if (num_pipes > 2)
		{
			double half_halo_cost = 0.0;
			for (int iGPU = _GPUs_per_pipe-1, i = 0;  iGPU >= 0;  --iGPU)
			{
				for (int iStep = num_timesteps*2-1;  iStep >= 0;  --iStep, ++i)
				{
					if (partial_allowed || iGPU == 0) half_halo_cost += (double)(i * half_stencil * _rel_cost[i&1]);
				}
			}
			//for (int i = 1;  i < _GPUs_per_pipe*num_timesteps*2;  ++i) half_halo_cost += (double)(i * half_stencil * _rel_cost[i&1]);
			double extra_work = half_halo_cost / ((double)((partial_allowed ? _GPUs_per_pipe : 1) * num_timesteps * num_pipes) * 8.0);
			extra_work = 2.0 * extra_work / (_rel_cost[0] + _rel_cost[1]);
			double y0 = 0.0;
			for (int iPipe = 0;  iPipe < num_pipes;  ++iPipe)
			{
				double y1 = y0 + ((double)_ny/((double)num_pipes*8.0)) + ((iPipe == 0 || iPipe == (num_pipes-1)) ? extra_work * (double)(num_pipes - 2) / 2.0 : -extra_work);
				pipe_width[iPipe] = 8 * ((int)round(y1) - (int)round(y0));
				y0 = y1;
				if (pipe_width[iPipe] <= 0) load_balance_possible = false;
			}
			if (!load_balance_possible)
			{
				printf("Warning! Volume is too narrow for proper load balancing.\n");
				y0 = 0.0;
				for (int iPipe = 0;  iPipe < num_pipes;  ++iPipe)
				{
					double y1 = y0 + ((double)_ny/(double)num_pipes);
					pipe_width[iPipe] = (int)round(y1) - (int)round(y0);
					y0 = y1;
				}
			}
			
			double* cost = new double[num_pipes];
			double total_cost = 0.0;
			for (int iPipe = 0, y0 = 0;  iPipe < num_pipes;  y0+=pipe_width[iPipe++]) 
			{
				cost[iPipe] = Calculate_Cost(y0,pipe_width[iPipe],_ny,num_timesteps,_GPUs_per_pipe,half_stencil,_rel_cost);
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
			int k = (_ny + num_pipes - 1) / num_pipes;
			for (int iPipe = 0;  iPipe < num_pipes;  ++iPipe) pipe_width[iPipe] = k;
			pipe_width[num_pipes-1] -= k*num_pipes - _ny;
		}

		// ..between gpus within each pipe
		_pipes = new Elastic_Pipeline*[num_pipes];
		for (int iPipe = 0, pipe_y0 = 0;  iPipe < num_pipes;  pipe_y0+=pipe_width[iPipe++])
		{
			int** steps = new int*[_GPUs_per_pipe*num_timesteps*2];
			int y0 = pipe_y0;
			int y1 = y0 + pipe_width[iPipe] - 1;
			_pipes[iPipe] = new Elastic_Pipeline(_log_level,this,iPipe,y0,y1,0,_nz-1);
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
				if (y1 >= _ny) y1 = _ny-1;
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
					if (inp_b_y1 >= _ny) inp_b_y1 = _ny-1;
					tmp = new Elastic_Buffer(this,_pipes[iPipe],curr_device_id,(iStep+1)/2,((iStep+1)&1)==1,inp_b_y0,inp_b_y1,z0,z1,4,cbo,0L,0);
					_pipes[iPipe]->Append_Buffer(tmp);

					emcbo = cbo;
					cbo -= 2;
				}
				double local_cost = (double)ylen * _rel_cost[i&1];
				bool steps_per_gpu_exceeded = ((i+1)%(num_timesteps*2)) == 0 ? true : false;
				if ((partial_allowed && cost + local_cost >= max_cost_per_GPU) || (!partial_allowed && steps_per_gpu_exceeded))
				{
					if (partial_allowed && cost + local_cost > max_cost_per_GPU)
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
						if (inp_b_y1 >= _ny) inp_b_y1 = _ny-1;
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

						cost = 0; //cost + local_cost - max_cost_per_GPU;
						//printf("..GPU %d\n",iGPU+1);
						curr_device_id = device_id[iPipe*_GPUs_per_pipe+iGPU];

						// full step - receiving end
						tmp = new Elastic_Buffer(this,_pipes[iPipe],curr_device_id,(iStep+1)/2,((iStep+1)&1)==1,vals[0],vals[1],z0,z1,3,cbo,_pipes[iPipe]->Get_Buffer(-3),0);
						_pipes[iPipe]->Append_Buffer(tmp);

						int inp_b_y0 = vals[0] - half_stencil;
						int inp_b_y1 = vals[1] + half_stencil;
						if (inp_b_y0 < 0) inp_b_y0 = 0;
						if (inp_b_y1 >= _ny) inp_b_y1 = _ny-1;
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
	return load_balance_possible;
}

void Elastic_Propagator::Delete_Compute_Pipelines()
{
	if (_num_devices > 0)
	{
		for (int i = 0;  i < _num_devices;  ++i)
		{
			if (_tried_p2p[i] != 0L) 
			{
				for (int j = 0;  j < _num_devices;  ++j)
				{
					if (_tried_p2p[i][j] && i != j)
					{
						int yes_sir;
						gpuErrchk( cudaDeviceCanAccessPeer(&yes_sir, _device_id[i], _device_id[j]) );
						if (yes_sir)
						{
							cudaSetDevice(_device_id[i]);
							cudaDeviceDisablePeerAccess(_device_id[j]);
						}
					}
				}
				delete [] _tried_p2p[i];
			}
		}
		delete [] _tried_p2p;
		_tried_p2p = 0L;

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
			if (_rxx_streams[i] != 0L)
			{
				cudaStreamDestroy(_rxx_streams[i]);
				_rxx_streams[i] = 0L;
			}
		}
		delete [] _cmp_streams;
		delete [] _inp_streams;
		delete [] _out_streams;
		delete [] _rxx_streams;
		delete [] _device_id;
		_cmp_streams = 0L;
		_inp_streams = 0L;
		_out_streams = 0L;
		_rxx_streams = 0L;
		_device_id = 0L;
		_num_devices = 0;
	}
	if (_num_pipes > 0)
	{
		for (int iPipe = 0;  iPipe < _num_pipes;  ++iPipe)
		{
			delete _pipes[iPipe];
		}
		delete [] _pipes;
		_pipes = 0L;
		_num_pipes = 0;
	}
	_GPUs_per_pipe = 0;
}

void Elastic_Propagator::Automatically_Build_Compute_Pipelines()
{
	int cu_device_count;
        cudaGetDeviceCount(&cu_device_count);
        if (cu_device_count <= 0)
        {
                printf("No CUDA capable devices.\nExiting.\n");
                exit(-1);
        }

	_best_num_z = -1;
        int num_devices = _job->Get_Number_Of_GPU_Devices();
        if (num_devices <= 0)
        {
		// determine best fit for available hardware.
                printf("Automatic determination of best GPU configuration...\n");
		int done = false, done_steps = false;
		const int max_pipes = cu_device_count > 8 ? 8 : cu_device_count;
		const int max_steps = 6;
		int num_configs = max_pipes * (max_steps - 2);
		float* performance = new float[num_configs];
		int* perf_num_z = new int[num_configs];
		int** device_ids = new int*[num_configs];
		int* num_devices = new int[num_configs];
		for (int i = 0;  i < num_configs;  ++i)
		{
			performance[i] = 0.0f;
			perf_num_z[i] = -1;
			device_ids[i] = 0L;
			num_devices[i] = 0;
		}
		// set aside some of the GPU memory for miscellaneous items that will be allocated by each shot.
		const size_t misc_buffer_size = 50 * 1024 * 1024;
		float** misc_buffer = new float*[cu_device_count];
		for (int iDev = 0;  iDev < cu_device_count;  ++iDev)
		{
			cudaSetDevice(iDev);
			cudaMalloc((void**)(misc_buffer+iDev),misc_buffer_size);
		}
		printf("MAXIMUM PIPES = %d\n",max_pipes);
		for (int num_pipes = 1;  num_pipes <= max_pipes && !done;  num_pipes=num_pipes*2)
		{
			done_steps = false;
			// decide which parts are paged
			bool register_PV = false;
			bool register_ST = false;
			bool register_EM = false;
			if (num_pipes == 2)
			{
				register_PV = true;
			}
			else if (num_pipes > 2)
			{
				register_PV = true;
				register_ST = true;
				register_EM = true;
			}
			// set aside space for page tables
			float** page_tables_buffer = new float*[cu_device_count];
			size_t page_tables_buffer_size = 0;
			if (register_PV) page_tables_buffer_size += _NbX * (_blkSize_PV / 4096) * 12;
			if (register_ST) page_tables_buffer_size += _NbX * (_blkSize_ST / 4096) * 12;
			if (register_EM) page_tables_buffer_size += _NbX * (_blkSize_EM / 4096) * 12;
			for (int iDev = 0;  iDev < cu_device_count;  ++iDev)
			{
				if (page_tables_buffer_size == 0)
				{
					page_tables_buffer[iDev] = 0L;
				}
				else
				{
					cudaSetDevice(iDev);
					cudaMalloc((void**)(page_tables_buffer+iDev),page_tables_buffer_size);
				}
			}
			for (int num_steps = max_steps;  num_steps >= 3;  --num_steps)
			{
				int perf_idx = (num_pipes - 1) * (max_steps - 2) + (num_steps - 3);

				// determine maximum number of devices that can fit this configuration.
				int max_devices_per_pipe = cu_device_count / num_pipes;
				int nx = _job->Get_Propagation_NX();
				int bsX = _stencil_order / 2;
				int NbX = (nx + bsX - 1) / bsX;
				int num_devices_per_pipe = 1;
				for (int i = 2;  i <= max_devices_per_pipe;  ++i)
				{
					int min_blocks = (1+num_steps) * 2 * i + 2;
					if (min_blocks < NbX)
					{
						num_devices_per_pipe = i;
					}
					else
					{
						break;
					}
				}
				int num_devices_curr_conf = num_devices_per_pipe * num_pipes;
				num_devices[perf_idx] = num_devices_curr_conf;
				device_ids[perf_idx] = new int[num_devices_curr_conf];
				int num_pipes_first_socket = num_pipes - (num_pipes / 2);
				int num_devices_first_socket=0, num_devices_second_socket=0;
				if (num_pipes == 1 && num_devices_curr_conf > 8)
				{
					num_devices_second_socket = num_devices_curr_conf / 2;
					num_devices_first_socket = num_devices_curr_conf - num_devices_second_socket;
				}
				else
				{
					num_devices_first_socket = num_pipes_first_socket*num_devices_per_pipe;
					num_devices_second_socket = num_devices_curr_conf - num_devices_first_socket;
				}
				printf("#pipes = %d, #steps = %d :: #devices = %d, #devices_1st_socket=%d, #devices_2nd_socket=%d\n",num_pipes,num_steps,num_devices_curr_conf,num_devices_first_socket,num_devices_second_socket);
				for (int iDev = 0;  iDev < num_devices_first_socket;  ++iDev) device_ids[perf_idx][iDev] = iDev;
				for (int iDev = 0;  iDev < num_devices_second_socket;  ++iDev) device_ids[perf_idx][num_devices_first_socket+iDev] = cu_device_count - num_devices_second_socket + iDev;
				//for (int iDev = 0;  iDev < num_devices_curr_conf-1;  ++iDev) printf("%d, ",device_ids[perf_idx][iDev]);
				//printf("%d\n",device_ids[perf_idx][num_devices_curr_conf-1]);

				// clear CUDA errors
				cudaGetLastError();
				bool load_balance_possible = Build_Compute_Pipelines(num_pipes,num_steps,device_ids[perf_idx],num_devices[perf_idx],false);
				Elastic_Shot* shot = _job->Get_Shot_By_Index(0);
				if (!load_balance_possible)
				{
					printf("Skipping this configuration.\n");
					shot->Free_Trace_Resample_Buffers();
					Delete_Compute_Pipelines();
				}
				else if (!Allocate_Device_Memory())
				{
					//done_steps = true;
					printf("Not enough device memory for this configuration!\n");
					shot->Free_Trace_Resample_Buffers();
					Free_Device_Memory();
					Delete_Compute_Pipelines();
				}
				else
				{
					if (num_pipes >= 4) done = true; // 8 pipes is overwhelmingly likely to be slower than 4

					// run 3 blocks to gauge average throughput
					Prepare_For_Propagation(shot,false,true);

					// skip past lead-in
					int ts_out = -1;
					while (ts_out < 0) Propagate_One_Block(_num_timesteps, shot, false, false, false, false, false, ts_out);

					perf_num_z[perf_idx] = -1;
					double max_mcells_per_second = 0.0;
					for (_curr_num_z = 0;  _curr_num_z < _num_num_z;  _curr_num_z+=1)
					{
						struct timespec ts0;
						clock_gettime(CLOCK_REALTIME, &ts0);

						// do entire major iteration
						const int max_num_iter = 10;
						int num_iter = 0;
						while (num_iter < max_num_iter) {Propagate_One_Block(_num_timesteps, shot, true, false, !register_PV, !register_ST, !register_EM, ts_out); ++num_iter;}

						struct timespec ts1;
						clock_gettime(CLOCK_REALTIME, &ts1);
						double elapsed_time = (double)ts1.tv_sec + (double)ts1.tv_nsec * 1e-9 - (double)ts0.tv_sec - (double)ts0.tv_nsec * 1e-9;

						double mcells = (double)(4 * _job->Get_Propagation_NY() * _job->Get_Propagation_NZ()) * (double)num_iter * 1e-6 * (double)(num_devices_per_pipe * num_steps);
						double mcells_per_second = mcells / elapsed_time;
						if (mcells_per_second > max_mcells_per_second)
						{
							max_mcells_per_second = mcells_per_second;
							perf_num_z[perf_idx] = _num_z[_curr_num_z];
						}
						
						printf("  -> #z=%d :: %.0f MCells/s\n",_num_z[_curr_num_z],mcells_per_second);
					}

					Release_Resources_After_Propagation(shot);
					Free_Device_Memory();
					Delete_Compute_Pipelines();

					performance[perf_idx] = max_mcells_per_second;
					printf("Max throughput was %.0f MCells/s (#z=%d)\n",max_mcells_per_second,perf_num_z[perf_idx]);
				}
			}
			// release page table buffer
			for (int iDev = 0;  iDev < cu_device_count;  ++iDev)
			{
				if (page_tables_buffer_size != 0)
				{
					cudaSetDevice(iDev);
					cudaFree(page_tables_buffer[iDev]);
					page_tables_buffer[iDev] = 0L;
				}
			}
			delete [] page_tables_buffer;
		}
		// release miscellaneous buffer
		for (int iDev = 0;  iDev < cu_device_count;  ++iDev)
		{
			if (misc_buffer[iDev] != 0L)
			{
				cudaSetDevice(iDev);
				cudaFree(misc_buffer[iDev]);
				misc_buffer[iDev] = 0L;
			}
		}
		delete [] misc_buffer;
		// find configuration with lowest elapsed time
		printf("\nDone optimizing hardware configuration\n");
		float highest_throughput = 0.0f;
		int best_num_pipes = 0, best_num_steps = 0;
		int best_num_devices = 0;
		int* best_device_ids = 0L;
		for (int num_pipes = 1;  num_pipes <= max_pipes;  ++num_pipes)
		{
			for (int num_steps = 3;  num_steps <= max_steps;  ++num_steps)
			{
				int perf_idx = (num_pipes - 1) * (max_steps - 2) + (num_steps - 3);
				float throughput = performance[perf_idx];
				if (throughput > 0.0f)
				{
					printf("#pipes = %d, #steps = %d :: %.0f MCells/s\n",num_pipes,num_steps,throughput);
					if (highest_throughput == 0.0f || throughput > highest_throughput)
					{
						highest_throughput = throughput;
						best_num_pipes = num_pipes;
						best_num_steps = num_steps;
						_best_num_z = perf_num_z[perf_idx];
						best_num_devices = num_devices[perf_idx];
						best_device_ids = device_ids[perf_idx];
					}
				}
			}
		}
		if (highest_throughput <= 0.0f)
		{
			printf("UNABLE TO FIND CONFIGURATION THAT CAN FIT ON AVAILABLE HARDWARE.\nExiting\n");
			exit(-1);
		}
		else
		{
			printf("Best configuration was determined to be %d pipes with %d timesteps per device (#=%d).\nManaged to use %d/%d devices.\n",best_num_pipes,best_num_steps,_best_num_z,best_num_devices,cu_device_count);
			cudaGetLastError();
			Build_Compute_Pipelines(
					best_num_pipes,
					best_num_steps,
					best_device_ids,
					best_num_devices,
					false
					);
			Allocate_Device_Memory();
			for (int num_pipes = 1;  num_pipes <= max_pipes && !done;  ++num_pipes)
			{
				for (int num_steps = 3;  num_steps <= max_steps;  ++num_steps)
				{
					int perf_idx = (num_pipes - 1) * (max_steps - 2) + (num_steps - 3);
					if (device_ids[perf_idx] != 0L) delete [] device_ids[perf_idx];
				}
			}
			delete [] device_ids;
			delete [] num_devices;
			delete [] perf_num_z;
			delete [] performance;
		}
	}		
	else
	{
		// user defined configuration
		Build_Compute_Pipelines(
				_job->Get_Number_Of_GPU_Pipes(),
				_job->Get_Steps_Per_GPU(),
				_job->Get_GPU_Devices(),
				_job->Get_Number_Of_GPU_Devices(),
				false
				);
		Allocate_Device_Memory();
	}
}

size_t Elastic_Propagator::Get_Minimum_Free_GPU_Memory()
{
	size_t min_gpu_memory_left = 0;
	for (int iDev = 0;  iDev < _num_devices;  ++iDev)
	{
		int device_ID = _device_id[iDev];
		cudaSetDevice(device_ID);
		size_t free, total;
		cudaError_t err = cudaMemGetInfo(&free,&total);
		if (err == cudaSuccess && (min_gpu_memory_left == 0 || free < min_gpu_memory_left)) min_gpu_memory_left = free;
	}
	return min_gpu_memory_left;
}

void Elastic_Propagator::Register_As_Many_Pages_As_Possible(bool verbose, bool do_PV, bool do_ST, bool do_EM)
{
	size_t min_free_GPU_memory = Get_Minimum_Free_GPU_Memory();
	if (verbose)
	{
		float min_gpu_free_MB = (double)min_free_GPU_memory / 1048576.0;
		printf("Minimum free GPU memory is %.2f MB\n",min_gpu_free_MB);
	}

	_num_pinned_PV = 0;
	_num_pinned_ST = 0;
	_num_pinned_EM = 0;

	const size_t min_allowed_remaining_memory_GPU = 50 * 1024 * 1024;
	if (min_free_GPU_memory < min_allowed_remaining_memory_GPU) return;

	// pin as many of the host blocks as GPU memory permits.
	int failed_to_pin = false;
	for (int ibX = 0;  ibX < _NbX;  ++ibX)
	{
		if (!failed_to_pin)
		{
			if (_PV_pinned[ibX])
			{
				++_num_pinned_PV;
			}
			else if (do_PV)
			{
				cudaError_t err = cudaHostRegister((void*)(_PV[ibX]),_blkSize_PV,0);
				if (err == cudaSuccess)
				{
					_PV_pinned[ibX] = true;
					++_num_pinned_PV;
					failed_to_pin = Get_Minimum_Free_GPU_Memory() < min_allowed_remaining_memory_GPU ? true : false;
				}
				else
				{
					failed_to_pin = true;
				}
			}
		}
	}
	for (int ibX = 0;  ibX < _NbX;  ++ibX)
	{
		if (!failed_to_pin)
		{
			if (_ST_pinned[ibX])
			{
				++_num_pinned_ST;
			}
			else if (do_ST)
			{
				cudaError_t err = cudaHostRegister((void*)(_ST[ibX]),_blkSize_ST,0);
				if (err == cudaSuccess)
				{
					_ST_pinned[ibX] = true;
					++_num_pinned_ST;
					failed_to_pin = Get_Minimum_Free_GPU_Memory() < min_allowed_remaining_memory_GPU ? true : false;
				}
				else
				{
					failed_to_pin = true;
				}
			}
		}
	}
	for (int ibX = 0;  ibX < _NbX;  ++ibX)
	{
		if (!failed_to_pin)
		{
			if (_EM_pinned[ibX])
			{
				++_num_pinned_EM;
			}
			else if (do_EM)
			{
				cudaError_t err = cudaHostRegister((void*)(_EM[ibX]),_blkSize_EM,0);
				if (err == cudaSuccess)
				{
					_EM_pinned[ibX] = true;
					++_num_pinned_EM;
					failed_to_pin = Get_Minimum_Free_GPU_Memory() < min_allowed_remaining_memory_GPU ? true : false;
				}
				else
				{
					failed_to_pin = true;
				}
			}
		}
	}

	if (verbose)
	{
		printf("Pinned %d/%d PV Host blocks.\n",_num_pinned_PV,_NbX);
		printf("Pinned %d/%d ST Host blocks.\n",_num_pinned_ST,_NbX);
		printf("Pinned %d/%d EM Host blocks.\n",_num_pinned_EM,_NbX);
	}
}

void Elastic_Propagator::Configure()
{
	Allocate_Host_Memory(false, false);
	Automatically_Build_Compute_Pipelines();
	Print_Graphical();
}

/*
Elastic_Propagator* Elastic_Propagator::Create_Best_Propagator_Configuration(Elastic_Modeling_Job* job)
{
	int cu_device_count;
	cudaGetDeviceCount(&cu_device_count);
	if (cu_device_count <= 0)
	{
		printf("No CUDA capable devices.\nExiting.\n");
		exit(-1);
	}

	int num_devices = job->Get_Number_Of_GPU_Devices();
	if (num_devices <= 0)
	{
		printf("Automatic determination of best GPU configuration...\n");
		int nx = job->Get_Propagation_NX();
		int Stencil_Order = 8;
		int bsX = Stencil_Order / 2;
		int NbX = (nx + bsX - 1) / bsX;
		int max_devices_per_pipe = 1;
		for (int i = 2;  i <= cu_device_count;  ++i)
		{
			int min_blocks = 8 * i + 2;
			if (min_blocks < NbX)
			{
				max_devices_per_pipe = i;
			}
			else
			{
				break;
			}
		}
		printf("Maximum devices that can fit in a single pipeline is %d.\n",max_devices_per_pipe);
		int* my_device_ids = new int[max_devices_per_pipe];
		for (int i = 0;  i < max_devices_per_pipe;  ++i) my_device_ids[i] = i;
		int max_steps_per_gpu = -1;
		job->Set_Number_Of_GPU_Pipes(1);
		job->Set_GPU_Devices(my_device_ids,max_devices_per_pipe);
		Elastic_Propagator* last_good_prop = 0L;
		for (int i = 3;  i < 10;  ++i)
		{
			int min_blocks = max_devices_per_pipe * 2 * (i + 1) + 1;
			if (min_blocks < NbX)
			{
				job->Set_Steps_Per_GPU(i);
				Elastic_Propagator* prop = new Elastic_Propagator(job);
				if (prop->Verify_All_Devices_Have_Enough_Memory())
				{
					delete last_good_prop;
					last_good_prop = prop;
					max_steps_per_gpu = i;
					printf("Able to fit %d timesteps per GPU.\n",i);
				}
				else
				{
					break;
				}
			}
			else
			{
				break;
			}
		}
		delete [] my_device_ids;
		if (max_steps_per_gpu > 0) job->Set_Steps_Per_GPU(max_steps_per_gpu);
		job->_propagator = last_good_prop;
		return last_good_prop;
	}
	else
	{
		return new Elastic_Propagator(job);
	}
}
*/

static void* thread_start(void* arg)
{
	Elastic_Propagator* prop = (Elastic_Propagator*)arg;
	/*
	if (prop->Get_Number_Of_Pipelines() == 2)
	{
		prop->Register_As_Many_Pages_As_Possible(false,true,false,false);
	}
	else if (prop->Get_Number_Of_Pipelines() > 2)
	{
		prop->Register_As_Many_Pages_As_Possible(false,true,true,true);
	}
	*/
	prop->Register_As_Many_Pages_As_Possible(false,true,true,true);
	return 0L;
}

void Elastic_Propagator::Read_Earth_Model()
{
	size_t min_free_GPU_memory = Get_Minimum_Free_GPU_Memory();
	float min_gpu_free_MB = (double)min_free_GPU_memory / 1048576.0;
	printf("Minimum free GPU memory is %.2f MB\n",min_gpu_free_MB);

	pthread_attr_t thrattr;
	int s = pthread_attr_init(&thrattr);
	if (s != 0)
	{
		printf("pthread_attr_init failed!\n");
		exit(-1);
	}
	s = pthread_attr_setstacksize(&thrattr, 16*1024*1024);
	if (s != 0)
        {
                printf("pthread_attr_setstacksize failed!\n");
                exit(-1);
        }
	pthread_t thread_id;
	s = pthread_create(&thread_id, &thrattr, &thread_start, this);
	if (s != 0)
	{
		printf("pthread_create failed!\n");
		exit(-1);
	}
	pthread_attr_destroy(&thrattr);

	_job->_Read_Earth_Model(this);

	void* res;
	s = pthread_join(thread_id, (void**)&res);
	if (s != 0L)
	{
		printf("pthread_join failed!\n");
		exit(-1);
	}
	if (_num_pinned_PV != 0 || _num_pinned_ST != 0 || _num_pinned_EM != 0)
	{
		printf("Pinned %d/%d PV Host blocks.\n",_num_pinned_PV,_NbX);
		printf("Pinned %d/%d ST Host blocks.\n",_num_pinned_ST,_NbX);
		printf("Pinned %d/%d EM Host blocks.\n",_num_pinned_EM,_NbX);
	}
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
	bool silent,
	unsigned int& word0,
        unsigned int& word1,
        unsigned int& word2,
        unsigned int& word3,
	bool& error
        )
{
        int xblk = x >> 2;
        int xidx = x & 3;

	error = false;
	if (xblk < 0 || xblk >= _NbX || y < 0 || y >= _ny || z < 0 || z >= _nz)
	{
		if (!silent)
		{
			printf("Elastic_Propagator::Get_EM_Cell - Out of bounds - x=%d,y=%d,z=%d\n",x,y,z);
			exit(-1);
		}
		else
		{
			error = true;
		}
	}
	if (!error)
	{
		int one_wf_size_f = 4 * _nz;
		int one_y_size_f = one_wf_size_f * 4;

		int idx = one_y_size_f * (int)y + (int)z * 4 + xidx;

		word0 = ((unsigned int*)_EM[xblk])[idx                ];
		word1 = ((unsigned int*)_EM[xblk])[idx+  one_wf_size_f];
		word2 = ((unsigned int*)_EM[xblk])[idx+2*one_wf_size_f];
		word3 = ((unsigned int*)_EM[xblk])[idx+3*one_wf_size_f];
		//printf("x-y-z=%d,%d,%d :: words=%d,%d,%d,%d\n",x,y,z,word0,word1,word2,word3);
	}
	else
	{
		word0 = 0;
		word1 = 0;
		word2 = 0;
		word3 = 0;
	}
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

unsigned int Elastic_Propagator::_Get_Earth_Model_Word(int widx, int x,int y,int z, bool silent, bool& error)
{
	unsigned int word[4];
	Get_EM_Cell(x,y,z,silent,word[0],word[1],word[2],word[3],error);
	return word[widx];
}

void Elastic_Propagator::_Set_Earth_Model_Word(int widx, int x,int y,int z, unsigned int new_word)
{
	unsigned int word[4];
	bool error;
        Get_EM_Cell(x,y,z,false,word[0],word[1],word[2],word[3],error);
	word[widx] = new_word;
	Set_EM_Cell(x,y,z,word[0],word[1],word[2],word[3]);
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
				bool error;
				Get_EM_Cell(x,y,z0,false,word0,word1,word2,word3,error);
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
				bool error;
				Get_EM_Cell(x,y,z1,false,word0,word1,word2,word3,error);
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
					bool error;
					Get_EM_Cell(x0,y,z,false,word0,word1,word2,word3,error);
					for (int x = 0;  x < x0;  ++x)
					{
						Set_EM_Cell(x,y,z,word0,word1,word2,word3);
					}
				}
				if (do_hi)
				{
					bool error;
					Get_EM_Cell(x1,y,z,false,word0,word1,word2,word3,error);
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
					bool error;
					Get_EM_Cell(x,y0,z,false,word0,word1,word2,word3,error);
					for (int y = 0;  y < y0;  ++y)
					{
						Set_EM_Cell(x,y,z,word0,word1,word2,word3);
					}
				}
				if (do_hi)
				{
					bool error;
					Get_EM_Cell(x,y1,z,false,word0,word1,word2,word3,error);
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
	bool Is_Pinned = _pinned || (Is_Model && _EM_pinned[bX]) || (!Is_Model && Is_PV && _PV_pinned[bX]) || (!Is_Model && !Is_PV && _ST_pinned[bX]);
	if (Is_Pinned)
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

cudaStream_t Elastic_Propagator::Get_Receiver_Stream(int device_id)
{
	int device_index = Get_Device_Index(device_id);
	if (device_index >= 0)
	{
		if (_rxx_streams[device_index] == 0L)
		{
			cudaSetDevice(device_id);
			cudaStreamCreate(&(_rxx_streams[device_index]));
		}
		return _rxx_streams[device_index];
	}
	printf("Warning! Get_Receiver_Stream returned nil!\n");
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

void Elastic_Propagator::Copy_To_Pinned_Buffer(bool copyPV, bool copyST, bool copyEM, int input_block_offset, int output_block_offset)
{
	// only do this if main volume is not pinned
	if (!_pinned)
	{
		if (input_block_offset >= 0)
		{
			if (copyPV && !_PV_pinned[input_block_offset])
			{
				void* src_PV = Get_Block(input_block_offset, false, true);
				omp_memcpy(_pbuf_PV[1], src_PV, _blkSize_PV);
			}
			//_Compare((char*)_pbuf_PV[1], (char*)src_PV, _blkSize_PV);
			//printf("input_block_offset=%d, src_PV=%p\n",input_block_offset,src_PV);

			if (copyST && !_ST_pinned[input_block_offset])
			{
				void* src_ST = Get_Block(input_block_offset, false, false);
				omp_memcpy(_pbuf_ST[1], src_ST, _blkSize_ST);
			}
			//_Compare((char*)_pbuf_ST[1], (char*)src_ST, _blkSize_ST);
			//_Find_Non_Zeros((char*)src_ST, _blkSize_ST);
			//_Find_Non_Zeros((char*)_pbuf_ST[1], _blkSize_ST);
			//printf("input_block_offset=%d, src_ST=%p\n",input_block_offset,src_ST);

			if (copyEM && !_EM_pinned[input_block_offset])
			{
				void* src_EM = Get_Block(input_block_offset, true, false);
				omp_memcpy(_pbuf_EM[1], src_EM, _blkSize_EM);
			}
			//_Compare((char*)_pbuf_EM[1], (char*)src_EM, _blkSize_EM);
			//printf("input_block_offset=%d, src_EM=%p\n",input_block_offset,src_EM);
		}

		if (output_block_offset >= 0)
		{
			if (copyPV && !_PV_pinned[output_block_offset])
			{
				void* dst_PV_Out = Get_Block(output_block_offset, false, true);
				omp_memcpy(dst_PV_Out, _pbuf_PV_Out[0], _blkSize_PV);
			}

			if (copyST && !_ST_pinned[output_block_offset])
			{
				void* dst_ST_Out = Get_Block(output_block_offset, false, false);
				omp_memcpy(dst_ST_Out, _pbuf_ST_Out[0], _blkSize_ST);
			}

			if (copyEM && _debug && !_EM_pinned[output_block_offset])
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

void Elastic_Propagator::Prepare_For_Propagation(Elastic_Shot* shot, bool debug_output_source_wavelet, bool is_profiling_run)
{
	int vol_nz = _job->Get_Propagation_NZ() / 8;
	if (_best_num_z <= 0)
	{
		_num_z = new int[vol_nz];
		_num_z_throughput = new float[vol_nz];
		_num_num_z = 0;
		for (int num_z = 2;  num_z < vol_nz;  ++num_z)
		{
			int z_per_block = (vol_nz + num_z - 1) / num_z;
			int z_remainder = vol_nz - (num_z-1) * z_per_block;
			if (z_remainder > 0 && z_per_block >= 4)
			{
				_num_z[_num_num_z] = num_z;
				_num_z_throughput[_num_num_z] = 0.0f;
				if (!is_profiling_run) printf("num_z = %d -> z_per_block = %d\n",_num_z[_num_num_z],z_per_block);
				++_num_num_z;
			}
		}
	}
	else
	{
		_num_z = new int[1];
		_num_z_throughput = new float[1];
		_num_num_z = 1;
		_num_z[0] = _best_num_z;
		_num_z_throughput[0] = 0.0f;
	}
	_curr_num_z = 0;

	const double courant_safe = 0.97;

	// determine internal timestepping
	// ..Courant# for O(2) time leap frog SG FD
	double courant = 1.0 / (sqrt(3.0) * (Elastic_Buffer::_C0 - Elastic_Buffer::_C1 + Elastic_Buffer::_C2 - Elastic_Buffer::_C3));
	courant = courant_safe * courant;  // drop Courant# by this safety factor
	if (!is_profiling_run) printf("Courant# = %f\n",courant);
	if (shot->Get_Ordertime() == 4) 
	{
		courant = courant*0.8/0.54;  // O(4) time Courant# for Omelyan symplectic
	}

	double dl_min = _job->Get_DX();
	if (_job->Get_DY() < dl_min) dl_min = _job->Get_DY();
	if (_job->Get_DZ() < dl_min) dl_min = _job->Get_DZ();
	if (!is_profiling_run && _log_level > 3) printf("Smallest cell size is %lf\n",dl_min);

	if (!is_profiling_run && _log_level > 3) printf("Determining max(Vp)...\n");
	float glob_max_Vp = 0.0f;
	float glob_min_Q = 1e36f, glob_max_Q = -1e36f;
	if (is_profiling_run)
	{
		// propagation results will not be used, so take shortcuts
		glob_max_Vp = _job->Get_Earth_Model_Attribute_Max(_job->Attr_Idx_Vp);
		glob_min_Q = _job->Get_Earth_Model_Attribute_Min(_job->Attr_Idx_Q);
		glob_max_Q = _job->Get_Earth_Model_Attribute_Max(_job->Attr_Idx_Q);
	}
	else
	{
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
#pragma omp critical
			{
				if (my_max_Vp > glob_max_Vp) glob_max_Vp = my_max_Vp;
				if (my_min_Q < glob_min_Q) glob_min_Q = my_min_Q;
				if (my_max_Q > glob_max_Q) glob_max_Q = my_max_Q;
			}
		}
	}
	if (!is_profiling_run) printf("Vp max = %lf\n",glob_max_Vp);
	if (!is_profiling_run) printf("Q=[%e,%e]\n",glob_min_Q,glob_max_Q);

	// find lowest output sample rate
	double output_sample_rate = shot->Get_SEGY_File_by_Index(0)->Get_Sample_Rate();
	for (int i = 1;  i < shot->Get_Number_Of_SEGY_Files();  ++i)
	{
		if (shot->Get_SEGY_File_by_Index(i)->Get_Sample_Rate() < output_sample_rate)
		{
			output_sample_rate = shot->Get_SEGY_File_by_Index(i)->Get_Sample_Rate();
		}
	}

	double dti_max = courant * dl_min / glob_max_Vp;
	if (!is_profiling_run && _log_level > 2) printf("Maximum internal dt for stability is %lfms\n",dti_max*1e3);
	double dti_courant_factor = dti_max * _job->Get_Courant_Factor();
	if (!is_profiling_run && _log_level > 2) printf("Internal dt adjusted by Courant factor is %lfms\n",dti_courant_factor*1e3);
	
	if (dti_courant_factor >= output_sample_rate)
	{
		// dti = output_sample_rate in this case
		_dti = dti_courant_factor;
	}
	else
	{
		// round dti so that output_sample_rate becomes a multiple of dti
		double multiple = ceil(output_sample_rate / dti_courant_factor);
		_dti = output_sample_rate / multiple;
	}
	if (!is_profiling_run && _log_level > 2) printf("Internal dt is %lfms\n",_dti*1e3);

	_num_timesteps = (int)ceil(shot->Get_Propagation_Time()/_dti);
	if (!is_profiling_run) printf("%d timesteps.\n",_num_timesteps);

	_pbuf_first_call = true;
	for (int iBlk = 0;  iBlk < _NbX;  ++iBlk)
	{
		omp_memclear(_PV[iBlk], _blkSize_PV);
		omp_memclear(_ST[iBlk], _blkSize_ST);
	}
	for (int iPipe = 0;  iPipe < _num_pipes;  ++iPipe)
	{
		_pipes[iPipe]->Allocate_RxLoc_Buffer(shot);
		_pipes[iPipe]->Reset();
	}
	shot->Prepare_Source_Wavelet(_dti,debug_output_source_wavelet);
	shot->Allocate_Pinned_Host_Memory(this);
	shot->Create_Trace_Resample_Buffers(this);
}

void Elastic_Propagator::Release_Resources_After_Propagation(Elastic_Shot* shot)
{
	delete [] _num_z;
	delete [] _num_z_throughput;
	_num_z = 0L;
	_num_z_throughput = 0L;
	_num_num_z = 0;
	_curr_num_z = 0;
	for (int iPipe = 0;  iPipe < _num_pipes;  ++iPipe) _pipes[iPipe]->Free_RxLoc_Buffer(shot);
	shot->Free_Pinned_Host_Memory(this);
	shot->Free_Trace_Resample_Buffers();
}

void Elastic_Propagator::Propagate_Shot(Elastic_Shot* shot, bool debug_output_source_wavelet, bool debug_output_xz_slices)
{
	Prepare_For_Propagation(shot,debug_output_source_wavelet,false);
	int ts_out;
	while (!Propagate_One_Block(_num_timesteps, shot, false, debug_output_xz_slices, true, true, true, ts_out));
	shot->Write_SEGY_Files();
	Release_Resources_After_Propagation(shot);
	printf("Finished Elastic_Propagator::Propagate_Shot\n");
}

// returns true if at least Number_Of_Timesteps timesteps have been completed
bool Elastic_Propagator::Propagate_One_Block(int Number_Of_Timesteps, Elastic_Shot* shot, bool silent, bool debug_output_xz_slices, bool copyPV, bool copyST, bool copyEM, int& ts_output)
{
#ifdef DETAILED_TIMING
	struct timespec ts0;
	clock_gettime(CLOCK_REALTIME, &ts0);
#endif

	if (_pbuf_first_call)
	{
		Copy_To_Pinned_Buffer(copyPV, copyST, copyEM, _pipes[0]->Get_Input_Block_Offset(0), -1);
		clock_gettime(CLOCK_REALTIME, &_before);
		_pbuf_first_call = false;
	}
	else
	{
		for (int i = 0;  i < _num_pipes;  ++i) _pipes[i]->Shift_Buffers();
	}
	Shift_Pinned_Buffer();

	// start data transfers.
	// launch longest running compute kernel on each GPU. This is always the first kernel in the pipe.
	Elastic_Buffer*** launch_buffers = new Elastic_Buffer**[_GPUs_per_pipe];
	if (_slow_data_transfers)
	{
		for (int iGPU = 0;  iGPU < _GPUs_per_pipe;  ++iGPU)
		{
			for (int iPipe = 0;  iPipe < _num_pipes;  ++iPipe)
			{
				int device_id = _pipes[iPipe]->Get_All_Device_IDs()[iGPU];
				cudaSetDevice(device_id);
				for (int iBuff = 0;  iBuff < _pipes[iPipe]->Get_Number_Of_Buffers();  ++iBuff)
				{
					Elastic_Buffer* buf = _pipes[iPipe]->Get_Buffer(iBuff);
					if (buf->Get_Device_ID() == device_id)
					{
						buf->Launch_Input_Transfers();
						buf->Launch_Output_Transfers();
					}
				}
			}
		}
		for (int iGPU = 0;  iGPU < _GPUs_per_pipe;  ++iGPU)
		{
			launch_buffers[iGPU] = new Elastic_Buffer*[_num_pipes];
			for (int iPipe = 0;  iPipe < _num_pipes;  ++iPipe)
			{
				int device_id = _pipes[iPipe]->Get_All_Device_IDs()[iGPU];
				cudaSetDevice(device_id);
				launch_buffers[iGPU][iPipe] = 0L;
				for (int iBuff = 0;  iBuff < _pipes[iPipe]->Get_Number_Of_Buffers();  ++iBuff)
				{
					Elastic_Buffer* buf = _pipes[iPipe]->Get_Buffer(iBuff);
					if (buf->Get_Device_ID() == device_id)
					{
						if (buf->Is_Compute() && !buf->Get_M1_Buffer()->Is_Compute())
						{	
							launch_buffers[iGPU][iPipe] = buf;
							buf->Launch_Compute_Kernel(false,_dti,shot,_num_z[_curr_num_z]);
						}
					}
				}
			}
		}
	}
	else
	{
		for (int iGPU = 0;  iGPU < _GPUs_per_pipe;  ++iGPU)
		{
			launch_buffers[iGPU] = new Elastic_Buffer*[_num_pipes];
			for (int iPipe = 0;  iPipe < _num_pipes;  ++iPipe)
			{
				int device_id = _pipes[iPipe]->Get_All_Device_IDs()[iGPU];
				cudaSetDevice(device_id);
				launch_buffers[iGPU][iPipe] = 0L;
				for (int iBuff = 0;  iBuff < _pipes[iPipe]->Get_Number_Of_Buffers();  ++iBuff)
				{
					Elastic_Buffer* buf = _pipes[iPipe]->Get_Buffer(iBuff);
					if (buf->Get_Device_ID() == device_id)
					{
						if (buf->Is_Compute() && !buf->Get_M1_Buffer()->Is_Compute())
						{	
							launch_buffers[iGPU][iPipe] = buf;
							buf->Launch_Compute_Kernel(false,_dti,shot,_num_z[_curr_num_z]);
						}
					}
				}
			}
		}
		for (int iGPU = 0;  iGPU < _GPUs_per_pipe;  ++iGPU)
		{
			for (int iPipe = 0;  iPipe < _num_pipes;  ++iPipe)
			{
				int device_id = _pipes[iPipe]->Get_All_Device_IDs()[iGPU];
				cudaSetDevice(device_id);
				for (int iBuff = 0;  iBuff < _pipes[iPipe]->Get_Number_Of_Buffers();  ++iBuff)
				{
					Elastic_Buffer* buf = _pipes[iPipe]->Get_Buffer(iBuff);
					if (buf->Get_Device_ID() == device_id)
					{
						buf->Launch_Input_Transfers();
						buf->Launch_Output_Transfers();
					}
				}
			}
		}
	}

	// launch remaining compute kernels.
	bool keep_doing_it = false;
	do
	{
		keep_doing_it = false;
		for (int iGPU = 0;  iGPU < _GPUs_per_pipe;  ++iGPU)
		{
			for (int iPipe = 0;  iPipe < _num_pipes;  ++iPipe)
			{
				int device_id = _pipes[iPipe]->Get_All_Device_IDs()[iGPU];
				cudaSetDevice(device_id);
				Elastic_Buffer* prev_compute_buf = launch_buffers[iGPU][iPipe];
				launch_buffers[iGPU][iPipe] = 0L;
				for (int iBuff = 0;  iBuff < _pipes[iPipe]->Get_Number_Of_Buffers();  ++iBuff)
				{
					Elastic_Buffer* buf = _pipes[iPipe]->Get_Buffer(iBuff);
					if (buf->Get_Device_ID() == device_id && buf->Get_M1_Buffer() == prev_compute_buf)
					{
						launch_buffers[iGPU][iPipe] = buf;
						buf->Launch_Compute_Kernel(false,_dti,shot,_num_z[_curr_num_z]);
						keep_doing_it = true;
					}
				}
			}
		}
	} while (keep_doing_it);
	
	// release temporary array(s)
	for (int iGPU = 0;  iGPU < _GPUs_per_pipe;  ++iGPU) delete [] launch_buffers[iGPU];
	delete [] launch_buffers;

	/*
	if (_slow_data_transfers)
	{
		for (int i = 0;  i < _num_pipes;  ++i) _pipes[i]->Launch_Data_Transfers();
		for (int i = 0;  i < _num_pipes;  ++i) _pipes[i]->Launch_Compute_Kernel(_dti,shot,_num_z[_curr_num_z]);
	}
	else
	{
		for (int i = 0;  i < _num_pipes;  ++i) _pipes[i]->Launch_Compute_Kernel(_dti,shot,_num_z[_curr_num_z]);
		for (int i = 0;  i < _num_pipes;  ++i) _pipes[i]->Launch_Data_Transfers();
	}
	*/
	for (int i = 0;  i < _num_pipes;  ++i) _pipes[i]->Launch_Receiver_Data_Transfers(shot);
	for (int i = 0;  i < _num_pipes;  ++i) _pipes[i]->Launch_Receiver_Extraction_Kernels(shot);

#ifdef DETAILED_TIMING
	struct timespec ts1;
	clock_gettime(CLOCK_REALTIME, &ts1);
#endif

	Copy_To_Pinned_Buffer(copyPV, copyST, copyEM, _pipes[0]->Get_Input_Block_Offset(1), _pipes[0]->Get_Output_Block_Offset(-1));

#ifdef DETAILED_TIMING
	struct timespec ts2;
	clock_gettime(CLOCK_REALTIME, &ts2);
#endif

	// demux receiver values from previous block
#pragma omp parallel for
	for (int iDev = 0;  iDev < _num_devices;  ++iDev)
	{
		// determine pipeline and device index
		bool done = false;
		for (int iPipe = 0, device_index_offset = 0;  !done && iPipe < _num_pipes;  ++iPipe)
		{
			int device_index = iDev - device_index_offset;
			if (device_index < _pipes[iPipe]->Get_Device_Count())
			{
				done = true;
				_pipes[iPipe]->DEMUX_Receiver_Values_For_One_Device(shot, device_index);
				//printf("DEMUX_Receiver_Values_One_Device(pipe=%d, device_index=%d, iDev=%d)\n",iPipe,device_index,iDev);
			}
			else
			{
				device_index_offset += _pipes[iPipe]->Get_Device_Count();
			}
		}
	}
	//printf("\n");
	//for (int i = 0;  i < _num_pipes;  ++i) _pipes[i]->DEMUX_Receiver_Values(shot);

#ifdef DETAILED_TIMING
	struct timespec ts3;
	clock_gettime(CLOCK_REALTIME, &ts3);
#endif

	// wait for cuda streams
	for (int i = 0;  i < _num_devices;  ++i) if (_cmp_streams[i] != 0L) cudaStreamSynchronize(_cmp_streams[i]);
#ifdef DETAILED_TIMING
        struct timespec ts4;
        clock_gettime(CLOCK_REALTIME, &ts4);
#endif
	for (int i = 0;  i < _num_devices;  ++i) if (_inp_streams[i] != 0L) cudaStreamSynchronize(_inp_streams[i]);
	for (int i = 0;  i < _num_devices;  ++i) if (_out_streams[i] != 0L) cudaStreamSynchronize(_out_streams[i]);
	for (int i = 0;  i < _num_devices;  ++i) if (_rxx_streams[i] != 0L) cudaStreamSynchronize(_rxx_streams[i]);

        gpuErrchk( cudaPeekAtLastError() );

#ifdef DETAILED_TIMING
	struct timespec ts5;
	clock_gettime(CLOCK_REALTIME, &ts5);

	_timer1 += (double)ts1.tv_sec + (double)ts1.tv_nsec * 1e-9 - (double)ts0.tv_sec - (double)ts0.tv_nsec * 1e-9;
	_timer2 += (double)ts2.tv_sec + (double)ts2.tv_nsec * 1e-9 - (double)ts1.tv_sec - (double)ts1.tv_nsec * 1e-9;
	_timer3 += (double)ts3.tv_sec + (double)ts3.tv_nsec * 1e-9 - (double)ts2.tv_sec - (double)ts2.tv_nsec * 1e-9;
	_timer4 += (double)ts4.tv_sec + (double)ts4.tv_nsec * 1e-9 - (double)ts3.tv_sec - (double)ts3.tv_nsec * 1e-9;
	_timer5 += (double)ts5.tv_sec + (double)ts5.tv_nsec * 1e-9 - (double)ts4.tv_sec - (double)ts4.tv_nsec * 1e-9;
#endif

	ts_output = _pipes[0]->Get_Output_Block_Timestep(0) - Get_Total_Number_Of_Timesteps();
	if ((_pinned && _pipes[0]->Get_Output_Block_Offset(1) == 0) || (!_pinned && _pipes[0]->Get_Output_Block_Offset(0) == 0))
	{
		// finished one iteration
		struct timespec after;
		clock_gettime(CLOCK_REALTIME, &after);
		double elapsed_time = (double)after.tv_sec + (double)after.tv_nsec * 1e-9 - (double)_before.tv_sec - (double)_before.tv_nsec * 1e-9;
		double mcells_per_s = (double)_nx * (double)_ny * (double)_nz * (double)Get_Total_Number_Of_Timesteps() * 1e-6 / elapsed_time;
		double h2d_GB_per_s = (double)(_h2d - _prev_h2d) / (1073741824.0 * elapsed_time);
		double d2h_GB_per_s = (double)(_d2h - _prev_d2h) / (1073741824.0 * elapsed_time);
		double h2h_GB_per_s = (double)(_h2h - _prev_h2h) / (1073741824.0 * elapsed_time);
		_prev_h2d = _h2d;
		_prev_d2h = _d2h;
		_prev_h2h = _h2h;
		double rt1 = 100.0 * _timer1 / elapsed_time;
		double rt2 = 100.0 * _timer2 / elapsed_time;
		double rt3 = 100.0 * _timer3 / elapsed_time;
		double rt4 = 100.0 * _timer4 / elapsed_time;
		double rt5 = 100.0 * _timer5 / elapsed_time;
		_timer1 = 0.0;
		_timer2 = 0.0;
		_timer3 = 0.0;
		_timer4 = 0.0;
		_timer5 = 0.0;
		if (ts_output == 0)
		{
			if (!silent) printf("LEAD-IN (filling up pipeline) took %.2f secs\n",elapsed_time);
		}
		else
		{
#ifdef DETAILED_TIMING
			_slow_data_transfers = rt5 > 1.0 ? true : false;
			if (!silent)
			{
				if (_num_num_z > 1)
				{
					printf("Timesteps %4d to %4d (#Z=%3d) :: %.2f secs - %.0f MC/s - H2D %.1f GB/s, D2H %.1f GB/s, H2H %.1f GB/s - %.0f+%.0f+%.0f+%.0f+%.0f=%.0f\n",ts_output-Get_Total_Number_Of_Timesteps()+1,ts_output,_num_z[_curr_num_z],elapsed_time,mcells_per_s,h2d_GB_per_s,d2h_GB_per_s,h2h_GB_per_s,rt1,rt2,rt3,rt4,rt5,rt1+rt2+rt3+rt4+rt5);
				}
				else
				{
					printf("Timesteps %4d to %4d :: %.2f secs - %.0f MC/s - H2D %.1f GB/s, D2H %.1f GB/s, H2H %.1f GB/s - %.0f+%.0f+%.0f+%.0f+%.0f=%.0f\n",ts_output-Get_Total_Number_Of_Timesteps()+1,ts_output,elapsed_time,mcells_per_s,h2d_GB_per_s,d2h_GB_per_s,h2h_GB_per_s,rt1,rt2,rt3,rt4,rt5,rt1+rt2+rt3+rt4+rt5);
				}
#else
				if (_num_num_z > 1)
				{
					printf("Timesteps %4d to %4d (#Z=%3d) :: %.2f secs - %.0f MC/s - H2D %.1f GB/s, D2H %.1f GB/s, H2H %.1f\n",ts_output-Get_Total_Number_Of_Timesteps()+1,ts_output,_num_z[_curr_num_z],elapsed_time,mcells_per_s,h2d_GB_per_s,d2h_GB_per_s,h2h_GB_per_s);
				}
				else
				{
					printf("Timesteps %4d to %4d :: %.2f secs - %.0f MC/s - H2D %.1f GB/s, D2H %.1f GB/s, H2H %.1f GB/s\n",ts_output-Get_Total_Number_Of_Timesteps()+1,ts_output,elapsed_time,mcells_per_s,h2d_GB_per_s,d2h_GB_per_s,h2h_GB_per_s);
				}
#endif
			}
			if (!silent && _num_num_z > 1)
			{
				_num_z_throughput[_curr_num_z] = mcells_per_s;
				if (_curr_num_z < _num_num_z-1)
				{
					++_curr_num_z;
				}
				else
				{
					// bubble sort throughput numbers and throw away bottom half
					bool go_on = false;
					do
					{
						go_on = false;
						for (int i = 0;  i < _num_num_z-1;  ++i)
						{
							if (_num_z_throughput[i+1] > _num_z_throughput[i])
							{
								float tmp = _num_z_throughput[i];
								_num_z_throughput[i] = _num_z_throughput[i+1];
								_num_z_throughput[i+1] = tmp;
								int itmp = _num_z[i];
								_num_z[i] = _num_z[i+1];
								_num_z[i+1] = itmp;
								go_on = true;
							}
						}
					} while (go_on);
					_num_num_z = _num_num_z / 2;
					_curr_num_z = 0;
				}
			}

			if (debug_output_xz_slices)
			{
				int iy = (int)round(shot->Get_Propagation_Source_Y());
				char path[4096];
				sprintf(path, "slices/xz_slice_Y=%04d_%04d_P",iy,ts_output);
				_job->Write_XZ_Slice(path, 3, iy);
				sprintf(path, "slices/xz_slice_Y=%04d_%04d_Vz",iy,ts_output);
				_job->Write_XZ_Slice(path, 2, iy);

				int iz = (int)round(shot->Get_Propagation_Source_Z());
				sprintf(path, "slices/xy_slice_Z=%04d_%04d_P",iz,ts_output);
				_job->Write_XY_Slice(path, 3, iz);
				sprintf(path, "slices/xy_slice_Z=%04d_%04d_Vz",iz,ts_output);
				_job->Write_XY_Slice(path, 2, iz);

				int ix = (int)round(shot->Get_Propagation_Source_X());
				sprintf(path, "slices/yz_slice_X=%04d_%04d_P",ix,ts_output);
				_job->Write_YZ_Slice(path, 3, ix);
				sprintf(path, "slices/yz_slice_X=%04d_%04d_Vz",ix,ts_output);
				_job->Write_YZ_Slice(path, 2, ix);

				/*
				   sprintf(path, "/panfs07/esdrd/tjhc/ELA_on_GPU/slices/xz_slice_Y=%04d_%04d_Vx",iy,ts);
				   _job->Write_XZ_Slice(path, 0, iy);

				   sprintf(path, "/panfs07/esdrd/tjhc/ELA_on_GPU/slices/xz_slice_Y=%04d_%04d_Vy",iy,ts);
				   _job->Write_XZ_Slice(path, 1, iy);

				   sprintf(path, "/panfs07/esdrd/tjhc/ELA_on_GPU/slices/xz_slice_Y=%04d_%04d_Vz",iy,ts);
				   _job->Write_XZ_Slice(path, 2, iy);
				 */
			}
		}
		clock_gettime(CLOCK_REALTIME, &_before);
		return ts_output >= Number_Of_Timesteps ? true : false;
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
				{
					if (_PV_pinned[i]) cudaHostUnregister(_PV[i]);
					free(_PV[i]);
				}
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
				{
					if (_ST_pinned[i]) cudaHostUnregister(_ST[i]);
					free(_ST[i]);
				}
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
				{
					if (_EM_pinned[i]) cudaHostUnregister(_EM[i]);
					free(_EM[i]);
				}
                                _EM[i] = 0L;
                        }
                }
                delete [] _EM;
                _EM = 0L;
        }
	delete [] _PV_pinned;
	_PV_pinned = 0L;
	delete [] _ST_pinned;
	_ST_pinned = 0L;
	delete [] _EM_pinned;
	_EM_pinned = 0L;
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
        for (size_t i = 0;  i < nn;  ++i)
        {
                __m128 zero = _mm_set_ps(0.0f, 0.0f, 0.0f, 0.0f);
                size_t i0 = (size_t)i * One_Thread_Full;
                size_t in = leni - i0;
                if (in > One_Thread_Full) in = One_Thread_Full;
                __m128* d = (__m128*)dst + i0;
                for (size_t j = 0;  j < in;  ++j)
                {
                        _mm_stream_ps((float*)(d+j), zero);
                }
        }
}

void Elastic_Propagator::omp_memcpy(void* dst, void* src, size_t len)
{
	Add_H2H(2*len);
        size_t leni = len / 16;
        size_t nn = (len + NUM_PAGES*getpagesize()-1) / (NUM_PAGES*getpagesize());
	size_t One_Thread_Full = NUM_PAGES * getpagesize() / sizeof(__m128);
#pragma omp parallel for schedule(static)
        for (size_t i = 0;  i < nn;  ++i)
        {
                size_t i0 = (size_t)i * One_Thread_Full;
                size_t in = leni - i0;
                if (in > One_Thread_Full) in = One_Thread_Full;
                __m128* d = (__m128*)dst + i0;
                __m128* s = (__m128*)src + i0;
                for (size_t j = 0;  j < in;  ++j)
                {
			if ((j&3) == 0) _mm_prefetch((char*)(s+j+16),_MM_HINT_T0);
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
	_PV_pinned = new bool[_NbX];
	_ST_pinned = new bool[_NbX];
	_EM_pinned = new bool[_NbX];
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
	// allocate high traffic memory blocks first.
	// the first memory that is allocated is faster than the last because of the way NUMA works.
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
	// allocate wavefield buffers next since they are both read and written
	for (int i = 0;  i < _NbX;  ++i)
	{
		if (Pinned)
		{
			cudaError_t err1 = cudaHostAlloc(&(_PV[i]),_blkSize_PV,cudaHostAllocDefault);
			if (err1 != cudaSuccess) _PV[i] = 0L;

			cudaError_t err2 = cudaHostAlloc(&(_ST[i]),_blkSize_ST,cudaHostAllocDefault);
			if (err2 != cudaSuccess) _ST[i] = 0L;
		}
		else
		{
			posix_memalign((void**)&(_PV[i]), getpagesize(), _blkSize_PV);
			omp_memclear(_PV[i], _blkSize_PV);
			_PV_pinned[i] = false;

			posix_memalign((void**)&(_ST[i]), getpagesize(), _blkSize_ST);
			omp_memclear(_ST[i], _blkSize_ST);
			_ST_pinned[i] = false;
		}
	}
	// allocate earth model buffers last since they are only read during propagation
	for (int i = 0;  i < _NbX;  ++i)
	{
		if (Pinned)
		{
			cudaError_t err3 = cudaHostAlloc(&(_EM[i]),_blkSize_EM,cudaHostAllocDefault);
			if (err3 != cudaSuccess) _EM[i] = 0L;
		}
		else
		{
			posix_memalign((void**)&(_EM[i]), getpagesize(), _blkSize_EM);
			omp_memclear(_EM[i], _blkSize_EM);
			_EM_pinned[i] = false;
		}
	}
	if (_debug)
	{
		// initialize with pattern, used for debugging
		long blkSize_PV_l = _blkSize_PV / 8;
		for (long blk = 0;  blk < _NbX;  ++blk)
		{
			for (long idx = 0;  idx < blkSize_PV_l;  ++idx)
			{
				((long**)_PV)[blk][idx] = blk * blkSize_PV_l + idx;
			}
		}

		long blkSize_ST_l = _blkSize_ST / 8;
                for (long blk = 0;  blk < _NbX;  ++blk)
                {
                        for (long idx = 0;  idx < blkSize_ST_l;  ++idx)
                        {
                                ((long**)_ST)[blk][idx] = blk * blkSize_ST_l + idx;
                        }
                }

		long blkSize_EM_l = _blkSize_EM / 8;
                for (long blk = 0;  blk < _NbX;  ++blk)
                {
                        for (long idx = 0;  idx < blkSize_EM_l;  ++idx)
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

		long blkSize_PV_l = _blkSize_PV / 8;
		for (long blk = 0;  blk < _NbX && !Error;  ++blk)
		{
			for (long idx = 0;  idx < blkSize_PV_l && !Error;  ++idx)
			{
				long val = blk * blkSize_PV_l + idx;
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

			long blkSize_ST_l = _blkSize_ST / 8;
			for (long blk = 0;  blk < _NbX && !Error;  ++blk)
			{
				for (long idx = 0;  idx < blkSize_ST_l && !Error;  ++idx)
				{
					long val = blk * blkSize_ST_l + idx;
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

				long blkSize_EM_l = _blkSize_EM / 8;
				for (long blk = 0;  blk < _NbX && !Error;  ++blk)
				{
					for (long idx = 0;  idx < blkSize_EM_l && !Error;  ++idx)
					{
						long val = blk * blkSize_EM_l + idx;
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

bool Elastic_Propagator::Allocate_Device_Memory()
{
	// no need to explicitly free device memory, each call to this function on the buffer objects frees its memory.
	/*
	for (int iDev = 0;  iDev < _num_devices;  ++iDev)
	{
		cudaSetDevice(_device_id[iDev]);
		printf("Resetting device %d\n",_device_id[iDev]);
		gpuErrchk( cudaDeviceReset() );
	}
	*/

	bool success = true;
	for (int i = 0;  i < _num_pipes && success;  ++i)
        {
                success = _pipes[i]->Allocate_Device_Memory();
        }
	return success;
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
		gpuErrchk( cudaDeviceCanAccessPeer(&yes_sir,device_id,peer_device_id) );
		if (yes_sir)
		{
			cudaSetDevice(device_id);
			gpuErrchk( cudaDeviceEnablePeerAccess(peer_device_id,0) );
			if (_log_level >= 4) printf("Enabled peer access for device %d to device %d\n",device_id,peer_device_id);
			return true;
		}
	}
	return false;
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

bool Elastic_Propagator::Verify_All_Devices_Have_Enough_Memory()
{
	for (int i = 0;  i < _num_pipes;  ++i)
	{
		if (!_pipes[i]->Verify_All_Devices_Have_Enough_Memory())
		{
			return false;
		}
	}
	return true;
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
				int Cores_per_SM = 0;
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
				if (_log_level >= 4) printf("device_id %d :: %s, CC=%d.%d, Free Mem=%.2f MB, %.3f TFLOPS, %.0f GB/s\n",device_id,devProps.name,devProps.major,devProps.minor,dFree_MB,TFLOPS,GB_per_s);
				return true;
			}
		}
		return false;
}

bool Elastic_Propagator::Check_GPUs(int* device_id, int num_devices)
{
	if (_log_level >= 4) printf("\n");
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
	if (_log_level >= 4) printf("Aggregate %.3f TFLOPS, %.0f GB/s\n\n",Total_TFLOPS,Total_GB_per_s);
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

void Elastic_Propagator::Add_H2H(unsigned long len)
{
	_h2h += len;
}

bool Elastic_Propagator::Is_Debug()
{
	return _debug;
}

