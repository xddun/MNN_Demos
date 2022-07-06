.SUFFIXES: .cpp .o

CC = g++

SRCS = ./main.cpp

OBJS = $(SRCS:.cpp=.o)

OUTPUT = main

OPENCV_ROOT=/Users/wanglei/opt/opencv-4.4.0/install_opencv
MNN_ROOT=/Users/wanglei/opt/MNN

CFLAGS = -I$(OPENCV_ROOT)/include/opencv4 \
		 -I$(MNN_ROOT)/include \
		 -I$(MNN_ROOT)/include/MNN \
		 -I$(MNN_ROOT)/3rd_party/imageHelper \
		 -DEO_USE_MNN

LIBS += -L$(OPENCV_ROOT)/lib -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_core \
        -L$(MNN_ROOT)/build_mnn -lMNN

all : $(OBJS)
	$(CC) -o $(OUTPUT) $(OBJS) $(LIBS)
	@echo "----- OK -----"

.cpp.o :
	$(CC) -O3 -std=c++11 -Wall $(CFLAGS) -o $@ -c $<

clean :
	-rm -f $(OBJS)
	-rm -f .core*
	-rm $(OUTPUT)



