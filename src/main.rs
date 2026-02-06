use llm_rs::tensors::Tensor;
use llm_rs::ops::matmul::MatMul;

fn main() {
    let a = Tensor::new(2, 3);
    let b = Tensor::new(3, 4);
    // let c = MatMul::forward(&a, &b);
    println!("c: {:?}", c);
}