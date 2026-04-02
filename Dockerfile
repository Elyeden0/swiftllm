FROM rust:1.77-bookworm AS builder
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY include/ include/
COPY dashboard/ dashboard/
RUN cargo build --release --bin swiftllm-server --bin swiftllm-mcp

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /src/target/release/swiftllm-server /usr/local/bin/
COPY --from=builder /src/target/release/swiftllm-mcp /usr/local/bin/
COPY --from=builder /src/dashboard/ /opt/swiftllm/dashboard/
EXPOSE 3000
ENV RUST_LOG=info
ENTRYPOINT ["swiftllm-server"]
