include ../support/Makefile.inc
include ../support/autoscheduler.inc
#CXX=$(CLANG)
HL_TARGET=host-cuda-cuda_capability_61-no_asserts
MATRIX_SIZE ?= 1024
TRANSPOSE ?= 0
all: $(BIN)/test
CUDA_SDK ?= /usr/local/cuda-10.0

CXXFLAGS += -I $(CUDA_SDK)/include
LDFLAGS += -L $(CUDA_SDK)/lib64


$(BIN)/mat_mul.generator: mat_mul_generator.cpp $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX)  $(CXXFLAGS) -g $(filter-out %.h,$^) -o $@ $(LDFLAGS) $(HALIDE_SYSTEM_LIBS) $(USE_EXPORT_DYNAMIC)


$(BIN)/mat_mul_manual.generator: mat_mul_generator_manual.cpp $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX)  $(CXXFLAGS) -g $(filter-out %.h,$^) -o $@ $(LDFLAGS) $(HALIDE_SYSTEM_LIBS) $(USE_EXPORT_DYNAMIC)

$(BIN)/transpose.generator: transpose.cpp $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX)  $(CXXFLAGS) -g $(filter-out %.h,$^) -o $@ $(LDFLAGS) $(HALIDE_SYSTEM_LIBS) $(USE_EXPORT_DYNAMIC)




$(BIN)/mat_mul_tensor.a: $(BIN)/mat_mul.generator
	@mkdir -p $(@D)
	$< -g mat_mul -f mat_mul_tensor -o $(BIN) target=$(HL_TARGET) use_tensor=true auto_schedule=false matrix_size=$(MATRIX_SIZE) -e h,static_library

$(BIN)/mat_mul.a: $(BIN)/mat_mul_manual.generator
	@mkdir -p $(@D)
	$< -g mat_mul -f mat_mul -o $(BIN) target=$(HL_TARGET)  auto_schedule=false  size=$(MATRIX_SIZE) -e h,static_library

$(BIN)/transpose.a: $(BIN)/transpose.generator
	@mkdir -p $(@D)
	$< -g transpose -f transpose -o $(BIN) target=$(HL_TARGET)  auto_schedule=false  matrix_size=$(MATRIX_SIZE) -e h,static_library


$(BIN)/filter: filter.cpp $(BIN)/mat_mul.a $(BIN)/mat_mul_tensor.a $(BIN)/transpose.a
	@mkdir -p $(@D)
	$(CXX)   $(CXXFLAGS) -I$(BIN) -I /home/ssioutas/Halide_shared_tensor/include  $(HALIDE_SYSTEM_LIBS) -Wall -O3 $^ -o $@ $(LDFLAGS) -I ../support $(IMAGE_IO_FLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS) $(OPENGL_LDFLAGS) -L ../../bin  -lHalide -lcudart -lcublas


clean:
	rm -rf $(BIN)

test: $(BIN)/filter
	LD_LIBRARY_PATH=../../bin/ HL_CUDA_JIT_MAX_REGISTERS=256 $(BIN)/filter $(MATRIX_SIZE)  $(TRANSPOSE)

