CC = gcc
CXX = g++
CFLAGS=-fopenmp -O3 -fPIC -mavx -g
LDFLAGS=-fopenmp -lm

OBJECTS=CvxCompress.o Wavelet_Transform_Slow.o Wavelet_Transform_Fast.o Run_Length_Encode_Slow.o Block_Copy.o Read_Raw_Volume.o

all: CvxCompress_Test CvxCompress_Test_Dyn Test_Compression Compress_SEAM_Basin Test_With_Generated_Input

libcvxcompress.so : $(OBJECTS)
	$(CXX) -shared $(LDFLAGS) -o libcvxcompress.so $(OBJECTS)

Wavelet_Transform_Fast.o: Wavelet_Transform_Fast.cpp Ds79_Base.cpp Us79_Base.cpp 
	$(CXX) -c $(CFLAGS) $< 

CvxCompress_Test: CvxCompress_Test.o $(OBJECTS)
	$(CXX) $(LDFLAGS) $(OBJECTS)  CvxCompress_Test.o -o CvxCompress_Test

CvxCompress_Test_Dyn: CvxCompress_Test.o libcvxcompress.so
	$(CXX) $(LDFLAGS) CvxCompress_Test.o  -L. -lcvxcompress -o $@

CvxCompress_GenCode: CvxCompress_GenCode.o CvxCompress.hxx Wavelet_Transform_Slow.o
	$(CXX) -O2 Wavelet_Transform_Slow.o  CvxCompress_GenCode.o -o CvxCompress_GenCode

Test_Compression: Test_Compression.o $(OBJECTS)
	$(CXX) $(LDFLAGS) $(OBJECTS)  Test_Compression.o -o Test_Compression

Ds79_Base.cpp Us79_Base.cpp: CvxCompress.hxx CvxCompress_GenCode
	./CvxCompress_GenCode

Compress_SEAM_Basin: Compress_SEAM_Basin.o libcvxcompress.so 
	$(CXX) $(LDFLAGS) $<  -L. -lcvxcompress  -o $@

Test_With_Generated_Input: Test_With_Generated_Input.o libcvxcompress.so 
	$(CXX) $(LDFLAGS) $<  -L. -lcvxcompress  -o $@

%.o: %.c
	$(CC) -c $(CFLAGS) $*.c

%.o: %.cpp
	$(CXX) -c $(CFLAGS) $*.cpp

clean:
	rm -f *.o
	rm -f libcvxcompress.so CvxCompress_Test CvxCompress_Test_Dyn CvxCompress_GenCode Test_Compression Compress_SEAM_Basin Test_With_Generated_Input
	rm -f Ds79_Base.cpp Us79_Base.cpp
