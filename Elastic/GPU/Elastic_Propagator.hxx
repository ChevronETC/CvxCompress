#ifndef CVX_SEISMOD_ELASTIC_PROPAGATOR
#define CVX_SEISMOD_ELASTIC_PROPAGATOR

#include <time.h>
#include <cuda_runtime_api.h>

class Elastic_Pipeline;
class Elastic_Modeling_Job;
class Elastic_Shot;

class Elastic_Propagator
{
public:
	Elastic_Propagator(Elastic_Modeling_Job* job);
	~Elastic_Propagator();	

	bool Is_Debug();

	Elastic_Modeling_Job* Get_Job();

	int Get_Total_Number_Of_Timesteps();

	int Get_Stencil_Order();

	// get X block size
	int Get_Block_Size_X();
	int Get_Number_Of_Blocks();

	int Get_NX() {return _nx;}
	int Get_NY() {return _ny;}
	int Get_NZ() {return _nz;}

	float Get_DX() {return _dx;}
	float Get_DY() {return _dy;}
	float Get_DZ() {return _dz;}

	double Get_Relative_Cost(bool Is_PV);
	double Get_Minimum_Workload();

	int Get_Number_Of_Pipelines();
	Elastic_Pipeline* Get_Pipeline(int pipe_idx);

	void Print_Graphical();

	void Allocate_Host_Memory(bool Pinned, bool Patterned);
	void Free_Host_Memory();
	bool Check_Host_Memory();

	void Allocate_Device_Memory();
	void Free_Device_Memory();

	void Read_Earth_Model();

	void Set_EM_Cell(
			int x,
			int y,
			int z,
			unsigned int word0,
			unsigned int word1,
			unsigned int word2,
			unsigned int word3
			);
	void Get_EM_Cell(
			int x,
			int y,
			int z,
			unsigned int& word0,
			unsigned int& word1,
			unsigned int& word2,
			unsigned int& word3
			);

	float Get_Receiver_Value(int wf_type, int ix, int iy, int iz);
	void Set_WF_Value(int wf_type, int ix, int iy, int iz, float val);

	// get a particular block from compute domain
	void* Get_Block(int bX, bool Is_Model, bool Is_PV);

	// get a host block used for CPU <-> GPU data transfers.
	// this is either the block returned by Get_Block if pinned memory is used
	// or it is a pinned memory buffer.
	void* Get_Host_Block(int bX, bool Is_Model, bool Is_PV, bool Is_Input);

	bool Enable_Peer_Access(int device_id, int peer_device_id);

	cudaStream_t Get_Compute_Stream(int device_id);
	cudaStream_t Get_Input_Stream(int device_id);
	cudaStream_t Get_Output_Stream(int device_id);

	void Prepare_For_Propagation(Elastic_Shot* shot);
	void Propagate_Shot(Elastic_Shot* shot);
	bool Propagate_One_Block(int Number_Of_Timesteps, Elastic_Shot* shot);

	void Add_H2D(unsigned long len);
	void Add_D2H(unsigned long len);

private:
	friend class Elastic_Modeling_Job;

	Elastic_Modeling_Job* _job;
	int _log_level;

	void _init(
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
		  );

	void _Compare(char* src, char* dst, size_t len);
	void _Find_Non_Zeros(char* dst, size_t len);

	// calculate a number representative of total computational cost of all pipelines.
	double Calculate_Cost(int y0, int ylen, int ny, int num_timesteps, int GPUs_per_pipe, int half_stencil, double* rel_cost);
	
	bool Print_Device_Stats(int device_id, double& TFLOPS, double& GB_per_s);
	bool Check_GPUs(int* device_id, int num_devices);

	// get the index of device with this device_id
	int Get_Device_Index(int device_id);

	int* _device_id;
	int _num_devices;

	int _stencil_order;

	double _dti;
	int _num_timesteps;

	float _dx;		// X cell size
	float _dy;		// Y cell size
	float _dz;		// Z cell size

	int _nx;		// propagation grid dimensions
	int _ny;
	int _nz;
	
	bool _debug;		// flag indicating if this is a debug session
	
	void _Insert_Earth_Model_Stripe(
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
			);
	unsigned int _Get_Earth_Model_Word(int widx, int x,int y,int z);

	void _NABC_TOP_Extend(int z0);
	void _NABC_BOT_Extend(int z1);
	void _NABC_SDX_Extend(int x0, int x1);
	void _NABC_SDY_Extend(int y0, int y1);

	void omp_memclear(void* dst, size_t len);
	void omp_memcpy(void* dst, void* src, size_t len);

	int _bsX;		// X block size
	int _NbX;		// number of blocks
	int* _ts;		// current timestep for each block
	bool _pinned;		// FLAG indicating if pinned memory was used
	void** _PV;		// Vx, Vy, Vz, Sx, Sy and Sz (latter 3 are memory variables)
	void** _ST;		// txx, tyy, tzz, txy, txz and tyz
	void** _EM;		// earth model (14 values packed into 16 bytes)
	void** _pbuf_PV;	// pinned memory buffer for _PV.
	void** _pbuf_ST;	// pinned memory buffer for _ST.
	void** _pbuf_EM;	// pinned memory buffer for the earth model.
	void** _pbuf_PV_Out;	// pinned memory buffer for _PV output.
	void** _pbuf_ST_Out;	// pinned memory buffer for _ST output.
	void** _pbuf_EM_Out;	// pinned memory buffer for _EM output. Used for debugging only.

	size_t _blkSize;	// block size in number of cells
	size_t _blkSize_PV;	// block size for PV buffer in bytes
	size_t _blkSize_ST;	// block size for ST buffer in bytes
	size_t _blkSize_EM;	// block size for EM buffer in bytes

	bool** _tried_p2p;	// tried_p2p[device][peer_device] is true if cudaCanAccessPeer has been called with these parameters.

	void Copy_To_Pinned_Buffer(int input_block_offset, int output_block_offset);
	void Shift_Pinned_Buffer();

	void cuda_host_memalign(void** p, size_t alignment, size_t len);
	void cuda_host_free(void* p);

	bool _pbuf_first_call;

	int _num_pipes;
	int _GPUs_per_pipe;

	double* _rel_cost;

	struct timespec _before;

	unsigned long _h2d;
	unsigned long _d2h;

	cudaStream_t* _cmp_streams;
	cudaStream_t* _inp_streams;
	cudaStream_t* _out_streams;

	// pipe, gpu, timestep, substep, variable
	Elastic_Pipeline** _pipes;
};

#endif
