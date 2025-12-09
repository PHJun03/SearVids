# 1. Build Stage
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential cmake git pkg-config \
    python3 python3-distutils \
    libasio-dev \
    libopencv-dev \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev \
    libcurl4-openssl-dev tzdata ffmpeg \
    nlohmann-json3-dev wget \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download ONNX Runtime (Linux x64) → /app/third_party/onnxruntime
ARG ORT_VERSION=1.16.3
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
    make -j$(nproc)

# 2. Runtime Stage
FROM ubuntu:22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libswscale5 libswresample3 libavcodec58 libavformat58 libavutil56 libavfilter7 \
    libcurl4 libopencv-core4.5d ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV LD_LIBRARY_PATH=/app/third_party/onnxruntime/lib:$LD_LIBRARY_PATH

COPY --from=builder /app/build/SearvidsServer /app/SearvidsServer
COPY --from=builder /app/third_party/onnxruntime /app/third_party/onnxruntime

RUN echo "/app/third_party/onnxruntime/lib" > /etc/ld.so.conf.d/onnxruntime.conf && ldconfig

EXPOSE 8080
CMD ["./SearvidsServer"]