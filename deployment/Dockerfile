# Builder Stage: This is where we compile the Go binary and prepare ONNX Runtime
FROM golang:1.23-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache git upx curl tar

# ONNX Runtime version
ARG ONNX_VERSION="1.23.0"

# Download and setup ONNX Runtime for Linux x64
RUN curl -L -o onnxruntime.tgz \
    "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz" && \
    tar -xzf onnxruntime.tgz && \
    mv onnxruntime-linux-x64-${ONNX_VERSION} onnxruntime && \
    rm onnxruntime.tgz

# Copy Go module files to download the project's dependencies
COPY go.mod go.sum ./

# Download Go dependencies
RUN go mod download

# Copy the source code into the /app directory
COPY ./cmd ./cmd
COPY ./internal ./internal

# Build the Go application with version information passed as a build argument
ARG VERSION="v0.0.0"
ARG COMMIT="unknown"
ARG BUILD_DATE=""

# Set CGO environment variables for ONNX Runtime
ENV CGO_CFLAGS="-I/app/onnxruntime/include"
ENV CGO_LDFLAGS="-L/app/onnxruntime/lib -lonnxruntime"
ENV LD_LIBRARY_PATH="/app/onnxruntime/lib:$LD_LIBRARY_PATH"

RUN go build \
    -ldflags "-s -w -X github.com/MeKo-Tech/pogo/internal/version.Version=$VERSION -X github.com/MeKo-Tech/pogo/internal/version.GitCommit=$COMMIT -X github.com/MeKo-Tech/pogo/internal/version.BuildDate=$BUILD_DATE" \
    -o /pogo ./cmd/ocr

# Compress the resulting binary using upx to reduce its size
RUN upx /pogo


# Runner Stage: This is where we create the final container to run the application
FROM alpine:3.21 AS runner

# Install runtime dependencies
RUN apk add --no-cache tzdata ca-certificates && \
    cp /usr/share/zoneinfo/Europe/Berlin /etc/localtime && \
    echo "Europe/Berlin" > /etc/timezone

# Set timezone to Europe/Berlin
ENV TZ=Europe/Berlin

# Create a non-root user with a specific numeric UID and GID
RUN addgroup -g 1001 -S nonroot && adduser -u 1001 -S nonroot -G nonroot

WORKDIR /

# Copy ONNX Runtime libraries from builder
COPY --from=builder /app/onnxruntime/lib /usr/local/lib/
RUN ldconfig /usr/local/lib

# Set up models directory and copy models
ENV POGO_MODELS_DIR=/usr/share/pogo/models
COPY models /usr/share/pogo/models

# Copy the binary from builder stage
COPY --from=builder /pogo /bin/pogo

# Change ownership of necessary directories to the non-root user
RUN chown -R nonroot:nonroot /usr/share/pogo

# Switch to the non-root user for running the application, improving security
USER 1001

# Expose the default server port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /bin/pogo test || exit 1

# Specify the command to run the Go application when the container starts
ENTRYPOINT ["/bin/pogo"]
CMD ["serve", "--host", "0.0.0.0"]