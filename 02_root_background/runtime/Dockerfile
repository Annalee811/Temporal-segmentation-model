FROM tensorflow/tensorflow@sha256:1658b00f06cdf8316cd8a905391235dad4bf25a488f1ea989a98a9fe9ec0386e
# Download this official PyPI wheel on the host with TLS verification and verify SHA256:
# intel_tensorflow-1.12.0-cp35-cp35m-manylinux1_x86_64.whl
# 026ddaf85cc6b6c83e8ba153f0ba9405f1249f92e877cf307528a63cdc9828d2
COPY intel_tensorflow-1.12.0-cp35-cp35m-manylinux1_x86_64.whl /tmp/
RUN python -m pip install --no-index --disable-pip-version-check --no-deps --ignore-installed /tmp/intel_tensorflow-1.12.0-cp35-cp35m-manylinux1_x86_64.whl && rm /tmp/intel_tensorflow-1.12.0-cp35-cp35m-manylinux1_x86_64.whl
ENV OMP_NUM_THREADS=4 KMP_BLOCKTIME=0 TF_CPP_MIN_LOG_LEVEL=2
