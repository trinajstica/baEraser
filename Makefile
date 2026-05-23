# baEraser – Makefile
# Requires: GTK4, libadwaita-1, opencv4, onnxruntime

CXX      := g++
TARGET   := baEraser
SRCDIR   := src
BUILDDIR := build

# ONNX Runtime (bundled under third_party/)
ORT_DIR  := third_party/onnxruntime-linux-x64-1.26.0

# Sources
SRCS := $(wildcard $(SRCDIR)/*.cpp)
OBJS := $(SRCS:$(SRCDIR)/%.cpp=$(BUILDDIR)/%.o)

# pkg-config flags
GTK_CFLAGS  := $(shell pkg-config --cflags gtk4 libadwaita-1)
GTK_LIBS    := $(shell pkg-config --libs   gtk4 libadwaita-1)
CV_CFLAGS   := $(shell pkg-config --cflags opencv4)
# Only link the OpenCV modules we actually use — avoids pulling in
# highgui (Qt6) and videoio (GStreamer) which are not needed.
CV_LIBS     := -lopencv_photo -lopencv_imgcodecs -lopencv_imgproc \
               -lopencv_dnn -lopencv_core

# ORT flags
ORT_CFLAGS  := -I$(ORT_DIR)/include
ORT_LIBS    := -L$(ORT_DIR)/lib -lonnxruntime \
               -Wl,-rpath,'$$ORIGIN/../third_party/onnxruntime-linux-x64-1.26.0/lib'

# Compiler flags
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra \
            -Wno-unused-parameter \
            $(GTK_CFLAGS) $(CV_CFLAGS) $(ORT_CFLAGS)

LDFLAGS  := $(GTK_LIBS) $(CV_LIBS) $(ORT_LIBS) -lstdc++fs

.PHONY: all clean install run

all: $(BUILDDIR)/$(TARGET)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)
	@echo ""
	@echo "  Build uspešen: $(BUILDDIR)/$(TARGET)"
	@echo ""

clean:
	rm -rf $(BUILDDIR)

# Quick run
run: all
	./$(BUILDDIR)/$(TARGET)

# Install to /usr/local
install: all
	install -Dm755 $(BUILDDIR)/$(TARGET) /usr/local/bin/$(TARGET)
	@echo "Nameščeno v /usr/local/bin/$(TARGET)"
