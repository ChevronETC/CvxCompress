#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

#include "IO/Read_Raw_Volume.hxx"

void Read_Raw_Volume(const std::filesystem::path& data_path, int& nx, int& ny, int& nz, float*& vol)
{
	std::ifstream input_stream(data_path, std::ios::binary);
	if (input_stream)
	{
		bool read_header = (nx == 0 || ny == 0 || nz == 0);
		std::vector<std::uint64_t> header(3);

		input_stream.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header.size() * sizeof(std::uint64_t)));
		if (!input_stream)
		{
			printf("Error! Unable to read the volume header from %s.\nAborting\n", data_path.c_str());
			exit(-1);
		}

		if (read_header)
		{
			nz = static_cast<int>(header[0]);
			ny = static_cast<int>(header[1]);
			nx = static_cast<int>(header[2]);
		}

		size_t nn = (size_t)nx * (size_t)ny * (size_t)nz;
		posix_memalign((void**)&vol, 64, nn * (size_t)sizeof(float));

		input_stream.read(reinterpret_cast<char*>(vol), static_cast<std::streamsize>(nn * sizeof(float)));
		if (!input_stream)
		{
			printf("Error! Expected %lu floats in %s but failed while reading.\nAborting\n",(unsigned long)nn, data_path.c_str());
			exit(-1);
		}
	}
	else
	{
		printf("Error! Unable to open file %s for reading.\nAborting\n", data_path.c_str());
		exit(-1);
	}
}
