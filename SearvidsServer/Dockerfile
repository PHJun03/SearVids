# 1. Build Stage
FROM nvidia/cuda:12.2.0-base-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential cmake git pkg-config \
    python3 python3-distutils python3-pip \
    libasio-dev \
    libopencv-dev \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev \
    libcurl4-openssl-dev tzdata ffmpeg \
    nlohmann-json3-dev wget libpqxx-dev \
    && pip install --no-cache-dir -U yt-dlp \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download ONNX Runtime (Linux x64) → /app/third_party/onnxruntime
ARG ORT_VERSION=1.23.2
WORKDIR /tmp
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz \
    && tar -xzf onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz \
    && mkdir -p /app/third_party/onnxruntime \
    && cp -r onnxruntime-linux-x64-gpu-${ORT_VERSION}/* /app/third_party/onnxruntime/ \
    && rm -rf onnxruntime-linux-x64-gpu-${ORT_VERSION}*

# Install Crow (C++ Microframework for Web)
RUN git clone --branch v1.2.0 https://github.com/CrowCpp/Crow.git /tmp/crow \
    && cd /tmp/crow \
    && mkdir build && cd build \
    && cmake .. -DCROW_BUILD_EXAMPLES=OFF -DCROW_BUILD_TESTS=OFF \
    && make install \
    && rm -rf /tmp/crow

# Copy source code
WORKDIR /app
COPY . .

# Build
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_TENSORRT=ON && \
    make -j$(nproc)

# Create TensorRT cache directory
RUN mkdir -p /app/build/bin/tensorrt_cache

# 2. Runtime Stage
# Use newer CUDA image that includes cuDNN 9 (Required by ORT 1.19+)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/app/models/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates python3 python3-pip \
    libswscale5 libswresample3 libavcodec58 libavformat58 libavutil56 libavfilter7 \
    libcurl4 libopencv-core4.5d libopencv-imgcodecs4.5d libopencv-imgproc4.5d ffmpeg libpqxx-6.4 \
    libnvinfer10 libnvonnxparsers10 libnvinfer-plugin10 \
    && pip install --no-cache-dir -U yt-dlp transformers faster-whisper sentencepiece flask \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV LD_LIBRARY_PATH=/app/third_party/onnxruntime/lib:$LD_LIBRARY_PATH

COPY --from=builder /app/build/SearvidsServer /app/SearvidsServer
COPY --from=builder /app/src/tokenizer.py /app/tokenizer.py
COPY --from=builder /app/src/whisper_ct2.py /app/whisper_ct2.py
COPY --from=builder /app/src/whisper_server.py /app/whisper_server.py
COPY --from=builder /app/third_party/onnxruntime /app/third_party/onnxruntime

RUN echo "/app/third_party/onnxruntime/lib" > /etc/ld.so.conf.d/onnxruntime.conf && ldconfig

EXPOSE 8080
CMD ["./SearvidsServer"]