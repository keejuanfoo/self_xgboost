CXX = g++
CXXFLAGS = \
	-std=c++17

SRCS = \
	data/model_data_structure.cc \
	data/predictions_data.cc \
	data/training_data.cc \
	model/decision_tree.cc \
	model/tree_node.cc \
	model/xgboost.cc \
	utils/math_utils.cc

OBJS = $(SRCS:.cc=.o)

NVCC = nvcc
NVCCFLAGS = \
	-std=c++17 \
	-arch=sm_75

CUDA_SRCS = \
	model/calculate_gain_helper.cu

CUDA_OBJS = $(CUDA_SRCS:.cu=.o)

TRAIN_SRC = train_xgboost.cc
PREDICT_SRC = use_model.cc

TRAIN_OBJ = $(TRAIN_SRC:.cc=.o)
PREDICT_OBJ = $(PREDICT_SRC:.cc=.o)

TARGETS = train_xgboost use_model

LDLIBS = -lpthread

all: $(TARGETS)

train_xgboost: $(OBJS) $(CUDA_OBJS) $(TRAIN_OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDLIBS)

use_model: $(OBJS) $(CUDA_OBJS) $(PREDICT_OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(CUDA_OBJS) $(TRAIN_OBJ) $(PREDICT_OBJ) $(TARGETS)
