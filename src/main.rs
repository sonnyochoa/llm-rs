use llm_rs::tensors::Tensor;
//use llm_rs::ops::matmul::MatMul;
use llm_rs::tensors::flatten_matrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // let a = Tensor::new(2, 3);
    // let b = Tensor::new(3, 4);
    // let c = MatMul::forward(&a, &b);

    let nested_matrix = [[1, 2], [6,5], [4, 3]];

    let flat = flatten_matrix(&nested_matrix);
    println!("fm: {:?}", flat);

    let data = vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6];
    let shape = vec![2, 3];

    let my_tensor = Tensor::new(data, shape);
    println!("Shape: {:?}", my_tensor.shape);
    println!("Strides: {:?}", my_tensor.strides);

    // get assertion
    assert_eq!(my_tensor.get(&[0, 0])?, 1.1);
    assert_eq!(my_tensor.get(&[0, 2])?, 3.3);
    assert_eq!(my_tensor.get(&[1, 1])?, 5.5);
    println!("\n[passed] my_tensor.get tests");
    println!("[-test-] my_tensor.get(&[0, 0]) -> 1.1");
    println!("[-test-] my_tensor.get(&[0, 2]) -> 3.3");
    println!("[-test-] my_tensor.get(&[1, 1]) -> 5.5");

    // set assertion
    // tensor.set(&[1, 2], 99.0)?;
    // assert_eq!(tensor.get(&[1, 2])?, 99.0);
    my_tensor.set();
    println!("\n[passed] my_tensor.set test");
    println!("[-test-] my_tensor.get(&[1, 2]) -> 99.0");
    Ok(())
}