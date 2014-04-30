#ifndef CVX_SEISMOD_ELASTIC_PIPELINE
#define CVX_SEISMOD_ELASTIC_PIPELINE

class Elastic_Buffer;
class Elastic_Propagator;
class Elastic_Shot;

class Elastic_Pipeline
{
public:
	Elastic_Pipeline(
		Elastic_Propagator* prop,
		int pipe_id,
		int pipe_y0,
		int pipe_y1,
		int pipe_z0,
		int pipe_z1
		);

	~Elastic_Pipeline();

	void Append_Buffer(Elastic_Buffer* new_buffer);
	void Add_EM_Buffer(Elastic_Buffer* new_buffer);
	
	int Get_Y0() {return _pipe_y0;}
	int Get_Y1() {return _pipe_y1;}
	int Get_Width() {return _pipe_y1 - _pipe_y0 + 1;}

	int Get_Number_Of_Buffers();	
	Elastic_Buffer* Get_Buffer(int index);

	// get the number of devices contributing to this pipeline
	int Get_Device_Count();

	int* Get_All_Device_IDs();

	// get the total number of timesteps a block is propagated by this pipeline
	int Get_Total_Number_Of_Timesteps();

	// get the total spatial shift for this pipeline.
	// note that this is a negative number.
	int Get_Total_Block_Offset();

	// Compute total memory requirement for one device
	unsigned long Compute_Device_Memory_Requirement(int device_id);

	// get the input and output block offsets for the pinned buffers for current, past or future iteration.
	// returns -1 if block offsets are not valid yet.
	int Get_Input_Block_Offset(int iteration);
	int Get_Output_Block_Offset(int iteration);

	int Get_Input_Block_Timestep(int iteration);
	int Get_Output_Block_Timestep(int iteration);

	// Get total workload of this pipeline
	double Get_Workload();
	
	// Get total workload of this pipeline without the halo overhead
	double Get_Minimum_Workload();

	// Get workload for one device in this pipeline
	double Get_Workload(int device_id);
	
	// Get computational overhead for one device in this pipeline.
	// A value of 1.0 means no overhead, 1.5 means 50% overhead.
	double Get_Computational_Overhead(int device_id);

	void Print_Graphical(int device_id);
	void Print_Graphical();

	void Allocate_Device_Memory();
	void Free_Device_Memory();

	void Launch_Data_Transfers();	
	void Launch_Simple_Copy_Kernel();
	void Launch_Compute_Kernel(float dti, Elastic_Shot* shot);

private:
	friend class Elastic_Propagator;
	Elastic_Propagator* _prop;

	int _pipe_id;
	int _pipe_y0;
	int _pipe_y1;
	int _pipe_z0;
	int _pipe_z1;

	int _num_buffers;
	Elastic_Buffer** _buffers;
	void Shift_Buffers();

	int _num_devices;
	int* _device_IDs;
	void** _d_Mem;

	// get a list containing the device IDs of every buffer.
	// each device ID appears only once in this list.
	void _Compile_Device_IDs();

	bool Block_Is_Output_By_Relative_Offset(Elastic_Buffer* buffer, int relative_block_offset);
};

#endif
