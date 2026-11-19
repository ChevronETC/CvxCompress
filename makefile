CC ?= gcc
CXX ?= g++

CFLAGS=-fopenmp -O3 -fPIC -g -Wno-unused-result
# Check if we're on an x86 architecture and if CC supports -mavx
ifeq ($(shell uname -m), x86_64)
    CFLAGS += -mavx
endif

LDFLAGS=-fopenmp -lm
TFLAG=-std=c++11

# Detect platform and set library extension accordingly
ifeq ($(shell uname), Darwin)
    LIB_EXT = dylib
	rflags = -install_name @rpath/libcvxcompress.dylib
else
    LIB_EXT = so
	rflags =
endif

OBJECTS=CvxCompress.o Wavelet_Transform_Slow.o Wavelet_Transform_Fast.o Run_Length_Encode_Slow.o Block_Copy.o Read_Raw_Volume.o

all: CvxCompress_Test CvxCompress_Test_Dyn Test_Compression Compress_SEAM_Basin Test_With_Generated_Input

lib: $(OBJECTS)
	$(CXX) -shared $(LDFLAGS) -o libcvxcompress.$(LIB_EXT) $(OBJECTS) $(rflags)

libcvxcompress.$(LIB_EXT) : $(OBJECTS)
	$(CXX) -shared $(LDFLAGS) -o libcvxcompress.$(LIB_EXT) $(OBJECTS)

Wavelet_Transform_Fast.o: Wavelet_Transform_Fast.cpp Ds79_Base.cpp Us79_Base.cpp
	$(CXX) -c $(CFLAGS) $<

CvxCompress_Test: CvxCompress_Test.o $(OBJECTS)
	$(CXX) $(LDFLAGS) $(TFLAG) $(OBJECTS)  CvxCompress_Test.o -o CvxCompress_Test

CvxCompress_Test_Dyn: CvxCompress_Test.o libcvxcompress.$(LIB_EXT)
	$(CXX) $(LDFLAGS) CvxCompress_Test.o  -L. -lcvxcompress -o $@

CvxCompress_GenCode: CvxCompress_GenCode.o CvxCompress.hxx Wavelet_Transform_Slow.o
	$(CXX) -O2 Wavelet_Transform_Slow.o  CvxCompress_GenCode.o -o CvxCompress_GenCode

Test_Compression: Test_Compression.o $(OBJECTS)
	$(CXX) $(LDFLAGS) $(TFLAG) $(OBJECTS)  Test_Compression.o -o Test_Compression

Ds79_Base.cpp Us79_Base.cpp: CvxCompress.hxx CvxCompress_GenCode
	./CvxCompress_GenCode

Compress_SEAM_Basin: Compress_SEAM_Basin.o libcvxcompress.$(LIB_EXT)
	$(CXX) $(LDFLAGS) $(TFLAG) $<  -L. -lcvxcompress  -o $@

Test_With_Generated_Input: Test_With_Generated_Input.o libcvxcompress.$(LIB_EXT)
	$(CXX) $(LDFLAGS) $(TFLAG) $<  -L. -lcvxcompress  -o $@

%.o: %.c
	$(CC) -c $(CFLAGS) $*.c

%.o: %.cpp
	$(CXX) -c $(CFLAGS) $(TFLAG) $*.cpp

# ---------------------------------------------------------------------------
# CUDA library and tests
# ---------------------------------------------------------------------------

NVCC ?= nvcc
NVCCFLAGS ?= -O3
CUDA_DIR = CvxCompress-CUDA
CUDA_BUILDDIR = $(BUILDDIR)/cvxcompress-cuda
CUDA_INCLUDES = -I$(CUDA_DIR)/include -I$(CUDA_DIR)/src
CUDA_OBJECTS = $(addprefix $(CUDA_DIR)/,src/CvxCompress_CUDA.o src/CvxCompress_CUDA_helpers.o \
	io/Read_Raw_Volume.o io/Write_Raw_Volume.o)

libcudacvxcompress.$(LIB_EXT): $(CUDA_OBJECTS)
	$(NVCC) -shared -o $@ $(CUDA_OBJECTS)

$(CUDA_DIR)/src/CvxCompress_CUDA.o: $(CUDA_DIR)/src/CvxCompress_CUDA_helpers.cuh \
	$(CUDA_DIR)/include/CvxCompress_CUDA.hxx $(wildcard $(CUDA_DIR)/include/Common/*.h)

$(CUDA_DIR)/src/CvxCompress_CUDA_helpers.o: $(addprefix $(CUDA_DIR)/src/,CvxCompress_CUDA_kernels.cu \
	Wavelet_Transform_16.cu Run_Length_Encoding.cu) $(CUDA_DIR)/include/Run_Length_Escape_Codes.hxx

$(CUDA_DIR)/src/%.o: $(CUDA_DIR)/src/%.cu
	$(NVCC) $(NVCCFLAGS) -std=c++17 -lineinfo --maxrregcount=64 -Xcompiler=-fPIC $(CUDA_INCLUDES) -c $< -o $@

$(CUDA_DIR)/io/%.o: $(CUDA_DIR)/io/%.cpp $(CUDA_DIR)/include/IO/%.hxx
	$(CXX) -O3 -std=c++17 -fPIC $(CUDA_INCLUDES) -c $< -o $@

.PHONY: cmake-cvxcompress-cuda test-cvxcompress-cuda
cmake-cvxcompress-cuda:
	cmake -S $(CUDA_DIR) -B $(CUDA_BUILDDIR) -DCVXCOMPRESS_CUDA_BUILD_TESTS=ON $(CUDA_CMAKE_FLAGS)
	cmake --build $(CUDA_BUILDDIR)

test-cvxcompress-cuda: cmake-cvxcompress-cuda
	ctest --test-dir $(CUDA_BUILDDIR) --output-on-failure

clean:
	rm -f *.o
	rm -f libcvxcompress.$(LIB_EXT) CvxCompress_Test CvxCompress_Test_Dyn CvxCompress_GenCode Test_Compression Compress_SEAM_Basin Test_With_Generated_Input
	rm -f Ds79_Base.cpp Us79_Base.cpp
	rm -f $(CUDA_OBJECTS) libcudacvxcompress.$(LIB_EXT)
	rm -rf $(BUILDDIR)
