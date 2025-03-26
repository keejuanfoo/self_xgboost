CXX = g++
CXXFLAGS = \
	-std=c++2b \
	-target arm64-apple-macos

SRCS = \
	data/model_data_structure.cc \
	data/predictions_data.cc \
	data/training_data.cc \
	model/decision_tree.cc \
	model/tree_node.cc \
	model/xgboost.cc \
	utils/math_utils.cc

OBJS = $(SRCS:.cc=.o)

TRAIN_SRC = train_xgboost.cc
PREDICT_SRC = use_model.cc

TRAIN_OBJ = $(TRAIN_SRC:.cc=.o)
PREDICT_OBJ = $(PREDICT_SRC:.cc=.o)

TARGETS = train_xgboost use_model

LDLIBS = -lpthread

all: $(TARGETS)

train_xgboost: $(OBJS) $(TRAIN_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

use_model: $(OBJS) $(PREDICT_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TRAIN_OBJ) $(PREDICT_OBJ) $(TARGETS)
