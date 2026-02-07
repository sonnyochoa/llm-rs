# Challenge 2: Tensor Indexing

## Concept

Tensors store data in a flat array but expose a multi-dimensional interface. The `get()` and `set()` methods bridge this gap using the stride formula:

```
flat_index = indices[0] * strides[0] + indices[1] * strides[1] + ... + indices[n] * strides[n]
```

Error handling is critical - out-of-bounds access should return meaningful errors, not crash.

---

## The Challenge

### Part A: Basic Indexing

Implement `get(&[usize]) -> Result<f32, TensorError>` and `set(&[usize], f32) -> Result<(), TensorError>`.

Your test cases to pass:

```rust
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let mut tensor = Tensor::new(data, vec![2, 3]);

assert_eq!(tensor.get(&[0, 0])?, 1.0);
assert_eq!(tensor.get(&[0, 2])?, 3.0);
assert_eq!(tensor.get(&[1, 1])?, 5.0);

tensor.set(&[1, 2], 99.0)?;
assert_eq!(tensor.get(&[1, 2])?, 99.0);
```

### Part B: Error Handling

Make these return appropriate errors:

```rust
tensor.get(&[5, 0]);      // row out of bounds
tensor.get(&[0, 10]);     // column out of bounds  
tensor.get(&[0]);         // wrong number of dimensions
tensor.get(&[0, 0, 0]);   // too many dimensions
```

### Part C (Bonus): 3D Tensor

Verify your solution works for arbitrary dimensions:

```rust
let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
let tensor = Tensor::new(data, vec![2, 3, 4]);

assert_eq!(tensor.get(&[0, 0, 0])?, 1.0);
assert_eq!(tensor.get(&[1, 2, 3])?, 24.0);
assert_eq!(tensor.get(&[1, 0, 0])?, 13.0);  // Can you figure out why?
```

---

## Guiding Questions

1. What fields should `TensorError` have? What error variants do you need?
2. How do you calculate `flat_index` from `indices` and `strides`?
3. Where should bounds checking happen - before or during index calculation?
4. Should `Tensor` be mutable, or should `set()` take `&mut self`?

---

## Research Pointers

- Look at how NumPy handles `IndexError` messages
- Consider: `thiserror` crate for ergonomic error types (optional)
- Your existing `compute_strides` logic is key - the index formula is the inverse

---

## Design Decisions Made

- **Error handling**: `Result<T, TensorError>` style
- **API style**: Slice-based indexing (`tensor.get(&[1, 2])`)

---

## Verification

Run your tests with:
```bash
cargo test
```

---

## Status

- [ ] Part A: Basic `get()` and `set()`
- [ ] Part B: Error handling with `TensorError`
- [ ] Part C: 3D tensor verification
