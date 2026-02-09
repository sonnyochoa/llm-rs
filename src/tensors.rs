use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
pub enum TensorError {
    #[error("Dimension mismatch: expected {expected} dimensions, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Index out of bounds: index {index} is out of bounds for dimension {dim} with size {size}")]
    IndexOutOfBounds { dim: usize, index: usize, size: usize },
}

// Tensor struct and core logic
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let strides = compute_strides(&shape);
        Tensor {
            data,
            shape,
            strides,
        }
    }
}

// Helper function: calculate the strides for Row-Major order
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut current_stride = 1;

    // Work backwards from the last dimension to the first
    println!("\n---- ---- ---- ----");
    println!("// shape.len(): {:?}\n", shape.len());
    for i in (0..shape.len()).rev() {
        println!("* i: {:?}", i);
        strides[i] = current_stride;
        println!("-> strides[{i}]: {:?}", strides[i]);
        current_stride *= shape[i];
        println!("<-> current_stride: {:?}", current_stride);
    }
    println!("---- ---- ---- ----\n");
    strides
}

impl Tensor {
    pub fn get(&self, indices: &[usize]) -> Result<f32, TensorError> {
        // Compute the flat index
        let mut flat_index = 0;
        for (i, &idx) in indices.iter().enumerate() {
            flat_index += idx * self.strides[i];
        }

        Ok(self.data[flat_index])
    }

    pub fn set(&self) {
        println!("\nset test");
    }
}
pub fn flatten_matrix<T, const N: usize>(matrix: &[[T; N]]) -> &[T] {
    matrix.as_flattened()
}