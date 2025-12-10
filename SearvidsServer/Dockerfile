# 1. Build Stage
FROM ubuntu:22.04 AS builder

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
    nlohmann-json3-dev wget \
    && pip install --no-cache-dir -U yt-dlp \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download ONNX Runtime (Linux x64) → /app/third_party/onnxruntime
ARG ORT_VERSION=1.23.2
WORKDIR /tmp
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz \
 && tar -xzf onnxruntime-linux-x64-${ORT_VERSION}.tgz \
 && mkdir -p /app/third_party/onnxruntime \
 && cp -r onnxruntime-linux-x64-${ORT_VERSION}/* /app/third_party/onnxruntime/ \
 && rm -rf onnxruntime-linux-x64-${ORT_VERSION}* 

# Copy source code
WORKDIR /app
COPY . .

# Build
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    find . -name "libwhisper.so*" -exec cp {} bin/ \; && \
    find . -name "libggml*.so*" -exec cp {} bin/ \;

# 2. Runtime Stage
FROM ubuntu:22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/app/models/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates python3 python3-pip \
    libswscale5 libswresample3 libavcodec58 libavformat58 libavutil56 libavfilter7 \
    libcurl4 libopencv-core4.5d ffmpeg \
    && pip install --no-cache-dir -U yt-dlp transformers \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV LD_LIBRARY_PATH=/app/third_party/onnxruntime/lib:$LD_LIBRARY_PATH

COPY --from=builder /app/build/SearvidsServer /app/SearvidsServer
COPY --from=builder /app/build/bin/whisper-cli /app/whisper-cli
COPY --from=builder /app/src/tokenizer.py /app/tokenizer.py
# Copy shared libraries for whisper
COPY --from=builder /app/build/bin/libwhisper.so* /usr/lib/
COPY --from=builder /app/build/bin/libggml*.so* /usr/lib/
COPY --from=builder /app/third_party/onnxruntime /app/third_party/onnxruntime

RUN echo "/app/third_party/onnxruntime/lib" > /etc/ld.so.conf.d/onnxruntime.conf && ldconfig

EXPOSE 8080
CMD ["./SearvidsServer"]