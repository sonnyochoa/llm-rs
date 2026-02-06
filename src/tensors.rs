// Tensor struct and core logic
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<i32>,
    strides: [i32; 5],
}

enum strides {
    2,
    1,
}