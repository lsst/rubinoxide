TARGET_DIR = pylib
WHEEL_DIR = target/wheels

UNAME_S := $(shell uname -s)

# Because we link again conda's openblass we need to tell conda test where
# to find the library when it runs. This is a bit different on mac and
# linux as the mac version of rust MUST run with the system libc++. This
# means it must be first in the library search path.
ifeq ($(UNAME_S),Darwin)
	LIB_PREFIX := DYLD_LIBRARY_PATH=/usr/lib:$(CONDA_PREFIX)/lib:${DYLD_LIBRARY_PATH}
else
	LIB_PREFIX := LD_LIBRARY_PATH=/usr/lib:$(CONDA_PREFIX)/lib:${LD_LIBRARY_PATH}
endif

all: build install

clean:
	rm -rf $(TARGET_DIR)/*
	rm -rf $(WHEEL_DIR)/*.whl

build: clean
	# Turning manylinux off for default builds as it can cause bundling issues
	# if we end up wanting it for pip packages or something it should be added
	# as a seperate make target.
	maturin build -r --manylinux off

install: build
	mkdir -p $(TARGET_DIR)
	pip install -qqq $(WHEEL_DIR)/*.whl --target=$(TARGET_DIR)

test: build install
	$(LIB_PREFIX) cargo test
	pytest tests

.PHONY: all build install clean
