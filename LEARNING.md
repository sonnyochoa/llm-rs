# Learning Log - llm-rs

A tinygrad-inspired journey to understanding GPU programming and building a production LLM framework.

## Resources

### Primary References
- **Book**: "Programming Massively Parallel Processors" by Hwu, Kirk, Hajj
- **[tinygrad](https://github.com/tinygrad/tinygrad)** - Architecture and design philosophy reference
- **[llama2.c](https://github.com/karpathy/llama2.c)** - Karpathy's minimal LLM inference engine
- **[CUDA matmul notes](src/ops/matmul.rs)** - Our internal CUDA programming model documentation

### Additional References
- [candle](https://github.com/huggingface/candle) - Rust ML framework by Hugging Face
- [burn](https://github.com/tracel-ai/burn) - Comprehensive deep learning framework in Rust
- [ggml](https://github.com/ggerganov/ggml) - Efficient tensor library for ML

## Project Goals

### Learning Phase (Current)
- Understand tensor operations from first principles
- Learn GPU programming concepts (memory hierarchy, parallelism, optimization)
- Build intuition for automatic differentiation
- Explore different compute backends (CPU → WebGPU → CUDA)

### Production Phase (Future)
- Clean reimplementation with production-quality architecture
- Support AMD (ROCm/HIP), NVIDIA (CUDA), and portable (WebGPU) backends
- Run real LLMs (GPT-2, LLaMA) efficiently
- Competitive performance with established frameworks

## Hardware Environment
- **Desktop**: 128 GB RAM
- **CPU**: AMD RYZEN AI MAX+ 395
- **GPU**: Radeon 8060S × 32
- **Target Support**: AMD (primary), NVIDIA CUDA (compatibility)

---

## Phase 1: CPU Foundation

### Current: Tensor Implementation
**Status**: Starting
**Date**: [In Progress]

#### Goals
- [ ] Fix compilation errors
- [ ] Implement basic Tensor struct with Vec<f32> storage
- [ ] Understand shape and stride calculations
- [ ] Implement element-wise operations (add, subtract, multiply)
- [ ] Implement matrix multiplication (naive)
- [ ] Write comprehensive tests

#### Design Decisions
_To be filled in as we make them..._

---

## Concepts Learned

### Tensors
_Notes on multi-dimensional arrays, memory layout, strides..._

### Matrix Operations
_Notes on different algorithms, complexity, optimization strategies..._

### Automatic Differentiation
_Notes on computation graphs, backpropagation, gradient flow..._

---

## Implementation Notes

### Code Organization
```
src/
├── tensors.rs    # Core Tensor struct and operations
├── ops.rs        # Operation trait definitions
├── ops/
│   └── matmul.rs # Matrix multiplication implementations
```

### Testing Strategy
_Notes on test coverage, gradient checking, numerical stability..._

---

## Performance Tracking

### Benchmarks
_Track performance improvements as we optimize..._

| Operation | Naive | Optimized | Notes |
|-----------|-------|-----------|-------|
| MatMul 128x128 | - | - | |
| Element-wise Add | - | - | |

---

## Questions & Thoughts

### Open Questions
- How should we handle broadcasting semantics?
- What error handling strategy makes sense?
- Should we use `f32` or support multiple dtypes from the start?

### Future Considerations
- Memory pooling strategy for GPU
- Kernel fusion opportunities
- Quantization approaches (INT8, INT4)

---

## Daily Log

### [Date] - Project Start
- Conducted technical analysis of existing codebase
- Identified compilation issues and missing implementations
- Created learning roadmap (5 phases: CPU → Optimized CPU → AMD GPU → CUDA → Production)
- Created this learning log

### [Date] - Tensor Implementation Begins
_Start logging your progress here..._

---

## Useful Commands

```bash
# Build and run
cargo run

# Run tests
cargo test

# Run benchmarks
cargo bench

# Check without building
cargo check

# Run with optimizations
cargo run --release
```

---

## References & Links

### Matrix Multiplication Resources
- [CUDA SGEMM optimization series](https://github.com/siboehm/SGEMM_CUDA)
- Stride calculation reference: [NumPy internals](https://numpy.org/doc/stable/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray)

### Rust ML Ecosystem
- [Are we learning yet?](https://www.arewelearningyet.com/)

---

_This document evolves as the project progresses. Update frequently!_
