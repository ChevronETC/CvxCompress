#ifndef WRITE_RAW_VOLUME_HXX
#define WRITE_RAW_VOLUME_HXX

#include <filesystem>

/**
 * @brief Write data with a compression ratio header
 * @param data_path Path to the data file
 * @param compression_ratio Compression ratio
 * @param comp_time Compression time
 * @param decomp_time Decompression time
 * @param vol Pointer to the volume data
 */
void Write_Raw_Volume(
	const std::filesystem::path& data_path,
	float compression_ratio,
	float comp_time,
	float decomp_time,
	int nx, int ny, int nz,
	const float* vol
);

/**
 * @brief Write data with a compression ratio header
 * @param data_path Path to the data file char pointer
 * @param compression_ratio Compression ratio
 * @param comp_time Compression time
 * @param decomp_time Decompression time
 * @param vol Pointer to the volume data
 */
inline void Write_Raw_Volume(
	const char* data_path,
	float compression_ratio,
	float comp_time,
	float decomp_time,
	int nx, int ny, int nz,
	const float* vol
)
{
	Write_Raw_Volume(std::filesystem::path(data_path), compression_ratio, comp_time, decomp_time, nx, ny, nz, vol);
}

#endif