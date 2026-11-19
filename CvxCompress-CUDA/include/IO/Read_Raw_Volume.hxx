#ifndef READ_RAW_VOLUME_HXX
#define READ_RAW_VOLUME_HXX

#include <filesystem>

/**
 * @brief Read data
 * @param data_path Path to the data file
 * @param nx Number of data points in the x dimension
 * @param ny Number of data points in the y dimension
 * @param nz Number of data points in the z dimension
 * @param vol Pointer to the volume data
 */
void Read_Raw_Volume(
	const std::filesystem::path& data_path,
	int& nx,
	int& ny,
	int& nz,
	float*& vol
);

/**
 * @brief Read data
 * @param data_path Path to the data file char pointer
 * @param nx Number of data points in the x dimension
 * @param ny Number of data points in the y dimension
 * @param nz Number of data points in the z dimension
 * @param vol Pointer to the volume data
 */
inline void Read_Raw_Volume(
	const char* data_path, 
	int& nx, 
	int& ny, 
	int& nz, 
	float*& vol
)
{
	Read_Raw_Volume(std::filesystem::path(data_path), nx, ny, nz, vol);
}

#endif

