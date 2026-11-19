#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "IO/Write_Raw_Volume.hxx"

void Write_Raw_Volume(
	const std::filesystem::path& filename,
	float compression_ratio,
	float comp_time,
	float decomp_time,
	int nx, int ny, int nz,
	const float* vol)
{
	std::filesystem::create_directories(filename.parent_path());
	std::ofstream output_stream(filename, std::ios::binary);
	if (output_stream)
	{
		output_stream.write(reinterpret_cast<const char*>(&compression_ratio), sizeof(float));
		output_stream.write(reinterpret_cast<const char*>(&comp_time), sizeof(float));
		output_stream.write(reinterpret_cast<const char*>(&decomp_time), sizeof(float));
		size_t nn = (size_t)520 * (size_t)464 * (size_t)256;
		output_stream.write(reinterpret_cast<const char*>(vol), static_cast<std::streamsize>(nn * sizeof(float)));
		if (!output_stream)
		{
			printf("Error! Failed while writing %s.\nAborting\n", filename.c_str());
			exit(-1);
		}
	}
	else
	{
		printf("Error! Unable to open file %s for writing.\nAborting\n", filename.c_str());
		exit(-1);
	}
}